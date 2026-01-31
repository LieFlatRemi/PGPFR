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

        # self.prompt_query = QueryFn(n_classes, config)
        #
        # self.prompt = Prompt(config)

        self.spatial_adapter = Adapter(in_dim=128, out_dim=64, dropout=0.0, init_option="lora",
                 adapter_scalar="0.5", adapter_layernorm_option="in")
        self.temporal_adapter = Adapter(in_dim=128, out_dim=64, dropout=0.0, init_option="lora",
                 adapter_scalar="0.5", adapter_layernorm_option="in")

        # self.cls = nn.Linear(128, n_classes)
        self.cls = self.generate_fc(128, config.first_split_size)

    def forward(self, x, cur_task=0, prompt=None):
        # input shape: [batch_size, time_len, joint_num, 3]
        # 32 8 22 3

        time_len = x.shape[1]
        joint_num = x.shape[2]

        # reshape x     32 176 3
        x = x.reshape(-1, time_len * joint_num, 3)

        # 32 176 128
        x = self.initial(x)

        reduce_sim = np.array([0.0])

        x = self.forward_feature(x)

        pred = self.cls(x)
        return pred, reduce_sim

    def forward_feature(self, x):
        # input [batch_size, 176, 128]

        # spatal
        x = self.spatial_att(x)

        # sp-adapter
        x = self.spatial_adapter(x)

        # temporal  # 32 176 128
        x = self.temporal_att(x)

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