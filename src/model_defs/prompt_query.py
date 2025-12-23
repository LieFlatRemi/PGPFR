from .prompt import Prompt
from .st_att_layer import *
import torch.nn as nn
import torch

class QueryFn(nn.Module):
    def __init__(self, n_classes, config):
        super(QueryFn, self).__init__()

        h_dim = 32
        h_num= 8

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

        # query layer
        self.query = nn.Linear(128, config.prompt.embed_dim)

    def forward(self, x):
        # input shape: [batch_size, time_len, joint_num, 3]
        # 32 8 22 3

        time_len = x.shape[1]
        joint_num = x.shape[2]

        # reshape x     32 176 3
        x = x.reshape(-1, time_len * joint_num, 3)

        # 32 176 128
        x = self.initial(x)
        # 32 128
        x = self.forward_feature(x)

        # query
        x_query = self.query(x)

        return x_query # 32 128

    def forward_feature(self, x):
        # input [batch_size, 176, 128]

        # spatal
        x = self.spatial_att(x)
        # temporal  # 32 176 128
        x = self.temporal_att(x)

        # mean
        x = x.sum(1) / x.shape[1]

        return x