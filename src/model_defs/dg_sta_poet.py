from .classifier import CosineContinualLinear, SimpleContinualLinear
from .dg_sta_BN import SpatioTemporalAttention as st_layer_bn
from .dg_sta import SpatioTemporalAttention as st_layer
from .prompt import Prompt
from .prompt_query import QueryFn
from .st_att_layer import *
import torch.nn as nn
import torch
from functools import partial

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

        st_layer = partial(st_layer_bn,
                           d_in=d_feat,
                           d_out=d_feat,
                           n_heads=n_heads,
                           d_head=d_head,
                           seq_len=seq_len,
                           n_joints=n_joints,
                           dropout=dp_rate,
                           )

        self.initial = nn.Sequential(
            nn.Linear(in_channels, d_feat),
            nn.ReLU(),
            LayerNorm(d_feat),
            # nn.BatchNorm1d(seq_len * n_joints),
            nn.Dropout(dp_rate),
        )
        self.spatial_att = st_layer(dom_type='spatial')
        self.temporal_att = st_layer(dom_type='temporal')

        # self.prompt_query = QueryFn(n_classes, config)

        self.prompt_query = nn.Linear(d_feat, d_feat)

        self.prompt = Prompt(config)

        # self.cls = nn.Linear(128, n_classes)
        self.cls = self.generate_fc(128, config.first_split_size)

    def forward(self, x, cur_task=0, prompt=None, train_mode=1):
        # input shape: [batch_size, time_len, joint_num, 3]
        # 32 8 22 3

        # query
        # x_query = self.prompt_query(x)

        time_len = x.shape[1]
        joint_num = x.shape[2]

        # # reshape x     32 176 3
        # x = x.reshape(-1, time_len * joint_num, 3)
        #
        # # 32 176 128
        # raw = self.initial(x)

        # x_query = self.forward_feature(raw)

        """
         先经过init 整合offset prompt 后 再整合为三维  
        """
        x = x.reshape(-1, time_len * joint_num, 3)
        # 32 176 128
        raw = self.initial(x)

        # Task 0: 不使用prompt，直接使用raw features
        # Task 1+: 使用prompt
        if cur_task == 0:
            # Task 0: 跳过prompt融合，直接使用raw features
            x = raw
            reduce_sim = None
        else:
            x_feat = self.forward_feature(raw)
            x_query = self.prompt_query(x_feat)
            # Task 1+: 与prompt的key计算cos相似
            res = self.prompt(x_embed=raw, cur_task=cur_task, cls_features=x_query, train_mode=train_mode)
            x = res['prompted_embedding']
            reduce_sim = res['reduce_sim']
            # 跟prompt求和后，感觉是需要归一化一下
            # temp_norm = nn.LayerNorm(self.feature_dim).to(torch.device("cuda:0"))
            # x = temp_norm(x)

        # 再重新经过backbone
        x = self.forward_feature(x)

        pred = self.cls(x)
        return pred, reduce_sim

    def forward_feature(self, x):
        # input [batch_size, 176, 128]

        # spatal
        x = self.spatial_att(x)
        # temporal  # 32 176 128
        x = self.temporal_att(x)

        # mean
        # x = x.sum(1) / x.shape[1]
        x = torch.mean(x, dim=1)

        return x

    def update_fc(self, nb_classes, freeze_old=True):
        if self.cls is None:
            self.cls = self.generate_fc(self.feature_dim, nb_classes)
        else:
            self.cls.update(nb_classes, freeze_old=freeze_old)

    def generate_fc(self, in_dim, out_dim):
        cls = CosineContinualLinear(in_dim, out_dim)
        return cls