from functools import partial
from .dg_sta_BN import SpatioTemporalAttention as st_layer_bn
from .adapter import Adapter
from .classifier import CosineContinualLinear, SimpleContinualLinear
from .prompt import Prompt
from .prompt_query import QueryFn
from .st_att_layer import *
import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self, n_classes, config):
        super(Model, self).__init__()

        d_head = config.d_head      # h_dim
        n_heads= config.n_heads       # h_num
        n_joints = config.n_joints
        seq_len = config.seq_len
        d_feat = config.d_feat
        in_channels = config.in_channels

        self.feature_dim = config.d_feat

        dp_rate = config.dropout

        self.spatial_att = ST_ATT_Layer(input_size=128, output_size=128, h_num=n_heads, h_dim=d_head, dp_rate=dp_rate,
                                        domain="spatial", time_len=8)

        self.temporal_att = ST_ATT_Layer(input_size=128, output_size=128, h_num=n_heads, h_dim=d_head, dp_rate=dp_rate,
                                         domain="temporal", time_len=8)

        self.initial = nn.Sequential(
            nn.Linear(in_channels, d_feat),
            nn.ReLU(),
            LayerNorm(d_feat),
            # nn.BatchNorm1d(seq_len * n_joints),
            nn.Dropout(dp_rate),
        )

        self.prompt_query = nn.Linear(d_feat, d_feat)

        self.prompt = Prompt(config)

        self.spatial_adapter = Adapter(in_dim=128, out_dim=64, dropout=0.1, init_option="lora",
                 adapter_scalar="0.25", adapter_layernorm_option="in")
        self.temporal_adapter = Adapter(in_dim=128, out_dim=64, dropout=0.1, init_option="lora",
                 adapter_scalar="0.25", adapter_layernorm_option="in")

        # self.cls = nn.Linear(128, n_classes)
        self.cls = self.generate_fc(128, config.first_split_size)

    def forward(self, x, cur_task=0, train_mode=1):
        # input shape: [batch_size, time_len, joint_num, 3]
        # 32 8 22 3

        time_len = x.shape[1]
        joint_num = x.shape[2]

        # reshape x     32 176 3
        x = x.reshape(-1, time_len * joint_num, 3)

        # 32 176 128
        raw = self.initial(x)

        # Task 0: 不使用prompt，adapter，直接使用raw features
        # Task 1+: 使用prompt + adapter
        if cur_task == 0:
            # Task 0: 跳过prompt融合，直接使用raw features
            x = raw
            reduce_sim = None
            x = self.forward_feature(x, cur_task)
        else:
            x_feat = self.forward_feature(raw)
            # x_query = self.prompt_query(x_feat)
            x_query = x_feat
            res = self.prompt(x_embed=raw, cur_task=cur_task, cls_features=x_query, train_mode=train_mode)
            prompt_s = res['spatial_prompt']
            prompt_t = res['temporal_prompt']
            reduce_sim = res['reduce_sim']
            # dual prefix-tuning
            x = self.forward_feature(raw, cur_task, prompt_s, prompt_t)

        pred = self.cls(x)
        return pred, reduce_sim

    def forward_feature(self, x, cur_task=0, prompt_s=None, prompt_t=None):
        # input [batch_size, 176, 128]

        if cur_task == 0:
            # spatial
            x = self.spatial_att(x)
            # temporal  # 32 176 128
            x = self.temporal_att(x)

        else:
            # spatial
            x = self.spatial_att(x, prompt_s)
            # sp-adapter
            x = self.spatial_adapter(x)
            # temporal  # 32 176 128
            x = self.temporal_att(x, prompt_t)
            # te-adapter
            x = self.temporal_adapter(x)

        # mean
        x = x.sum(1) / x.shape[1]

        return x

    def update_fc(self, nb_classes, freeze_old=True):
        if self.cls is None:
            self.cls = self.generate_fc(self.feature_dim, nb_classes)
        else:
            self.cls.update(nb_classes, freeze_old=freeze_old)

    def generate_fc(self, in_dim, out_dim):
        cls = CosineContinualLinear(in_dim, out_dim)
        return cls