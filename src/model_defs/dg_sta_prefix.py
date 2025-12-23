from .classifier import CosineContinualLinear, SimpleContinualLinear
from .prompt import Prompt
from .prompt_query import QueryFn
from .st_att_layer import *
import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self, n_classes, config):
        super(Model, self).__init__()

        h_dim = 32
        h_num= 8

        self.feature_dim = 128

        dp_rate = config.dropout

        self.initial = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            LayerNorm(128),
            nn.Dropout(dp_rate),
        )
        #input_size, h_num, h_dim, dp_rate, time_len, domain
        self.spatial_att = ST_ATT_Layer(input_size=128,output_size= 128, h_num=h_num, h_dim=h_dim, dp_rate=dp_rate, domain="spatial", time_len = 8)


        self.temporal_att = ST_ATT_Layer(input_size=128, output_size= 128,h_num=h_num, h_dim=h_dim, dp_rate=dp_rate, domain="temporal", time_len = 8)

        self.prompt_query = QueryFn(n_classes, config)

        self.prompt = Prompt(config)

        # self.cls = nn.Linear(128, n_classes)
        self.cls = self.generate_fc(128, config.first_split_size)

        self.query = nn.Linear(128, 128)

    def forward(self, x, cur_task=0, prompt=None):
        # input shape: [batch_size, time_len, joint_num, 3]
        # 32 8 22 3

        if cur_task > 0:
            # query
            x_query = self.prompt_query(x)

        time_len = x.shape[1]
        joint_num = x.shape[2]

        # reshape x     32 176 3
        x = x.reshape(-1, time_len * joint_num, 3)

        # 32 176 128
        raw = self.initial(x)

        if cur_task > 0:
            # 与prompt的key计算cos相似
            res = self.prompt(x_embed=raw, cur_task=cur_task, cls_features=x_query)
            x = res['prompted_embedding']
            prompt_s = res['spatial_prompt']
            prompt_t = res['temporal_prompt']
            reduce_sim = res['reduce_sim']
            print(res['selected_prompts_dict'])
        else:
            x = raw
            prompt_s = None
            prompt_t = None
            reduce_sim = torch.zeros(1, 1)



        # 再重新经过backbone
        x = self.forward_feature(x, prompt_s, prompt_t)

        pred = self.cls(x)
        return pred, reduce_sim

    def forward_feature(self, x, prompt_s=None, prompt_t=None):
        # input [batch_size, 176, 128]

        # spatal
        x = self.spatial_att(x, prompt_s)
        # temporal  # 32 176 128
        x = self.temporal_att(x, prompt_t)

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