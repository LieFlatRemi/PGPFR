import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
POET algorithm (without expansion) uses a fized sized prompt pool in most NTU RGB+R experiments. 

For random prompts w/ freezing, we are following a simpler protocol: we select top k-random prompts based on cosine-sim, 
and use new prompts at the end of sequence (less important positions)
- we do this at both train and test unlike in gestures where we let it select all top_k at test time.
'''


class Prompt(nn.Module):
    def __init__(self, config):
        super().__init__()

        print(f'\n\n prompt configuration: Prompt type {type}, pool size {config.prompt.pool_size}, '
              f'length {config.prompt.prompt_length}, topk {config.prompt.top_k}, '
              f'temporal pe {config.prompt.temporal_pe}, random_prompts {config.prompt.random_k},'
              f' freeze prompts: {config.prompt.freeze_prevtask_prompts}, '
              f'sorting {config.prompt.sort} \n\n')

        self.length = config.prompt.prompt_length
        self.embed_dim = config.prompt.embed_dim
        self.embed_dim_prompt = config.prompt.embed_dim_prompt
        self.prompt_pool = config.prompt.prompt_pool
        self.embedding_key = config.prompt.embedding_key
        self.prompt_init = config.prompt.prompt_init
        self.prompt_key = config.prompt.prompt_key
        self.top_k = config.prompt.top_k
        self.required_k = config.prompt.top_k
        self.pool_size = config.prompt.pool_size
        self.batchwise_prompt = config.prompt.batchwise_prompt
        self.prompt_type = config.prompt.prompt_type
        self.random_k = config.prompt.random_k
        self.random_prompts = None
        self.freeze_previous_task_prompts = config.prompt.freeze_prevtask_prompts
        self.random_expand = config.prompt.random_expand
        self.force_select_new = config.prompt.force_select_new
        self.stack_random_before = config.prompt.stack_random_before
        self.sort = config.prompt.sort

        self.decouple_prompting = config.prompt.decouple

        self.pool_dict = torch.zeros((self.pool_size)).cuda()

        if self.prompt_pool:
            prompt_pool_shape = (self.pool_size, self.length, self.embed_dim_prompt)
            if config.prompt.prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt)

        # learnable prompts
        key_shape = (self.pool_size, self.embed_dim)
        if config.prompt.prompt_key_init == 'uniform':
            self.prompt_key = nn.Parameter(torch.randn(key_shape))
            nn.init.uniform_(self.prompt_key)

        # revert to 1600 when running cross-attention experiment
        self.multihead_attn = nn.MultiheadAttention(self.embed_dim_prompt, 1, batch_first=True)

        print(f'\n\n checksss inside prompt file pool size {self.prompt.size()}, key size {self.prompt_key.size()}')


        # L2P 用于prompt和输入连接后转维度 176x2 -> 176
        self.concat_fc = nn.Linear(self.top_k * self.length * 2, self.top_k * self.length)


    def update_prompt(self):
        if self.random_k <= 0:
            return
        # 新task，扩展random_k个prompt
        device = self.prompt.device
        dtype_prompt = self.prompt.dtype
        dtype_key = self.prompt_key.dtype

        # prompt
        old_pool_size = self.pool_size
        print(f're-init pool size as {self.pool_size} + {self.random_k}')
        self.pool_size = self.pool_size + self.random_k
        new_prompt = nn.Parameter(torch.empty(
            self.pool_size, self.length, self.embed_dim_prompt,
            device=device, dtype=dtype_prompt
        ))
        with torch.no_grad():
            # 先拷贝旧 prompt
            new_prompt[:old_pool_size].copy_(self.prompt.data)
            # 末尾 random_k 个是新 task 的 prompt，随机初始化
            nn.init.uniform_(new_prompt[old_pool_size:])

        self.prompt = new_prompt

        # key
        key_shape = (self.pool_size, self.embed_dim)
        new_key = nn.Parameter(torch.empty(
            self.pool_size, self.embed_dim,
            device=device, dtype=dtype_key
        ))
        with torch.no_grad():
            new_key[:old_pool_size].copy_(self.prompt_key.data)
            # 计算已有 keys 的均值
            mean_key = self.prompt_key[:old_pool_size].mean(dim=0, keepdim=True)  # [1, embed_dim]
            # 用均值初始化新 task 的所有 random_k 个 keys
            new_key[old_pool_size:] = mean_key.expand(self.random_k, -1)  # [random_k, embed_dim]

        self.prompt_key = new_key

        self.pool_dict = torch.zeros((self.pool_size), device=device)


    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward(self, x_embed, prompt_mask=None, cur_task=0, cls_features=None, train_mode=1):
        '''
        cls_feat is the query function output, it will be used to curate promnpt.
        '''
        out = dict()
        device = x_embed.device
        self.pool_dict = torch.zeros((self.pool_size)).to(device)

        if self.prompt_pool:
            # 是否冻结过去task的prompt
            if self.freeze_previous_task_prompts and cur_task > 0 and train_mode == 1:
                previous_prompt_tracker = cur_task - 1
                prompt = torch.cat((self.prompt[:previous_prompt_tracker, :, :].detach(),
                                    self.prompt[previous_prompt_tracker:, :, :]), dim=0)
                prompt_key = self.prompt_key
            else:
                prompt = self.prompt
                prompt_key = self.prompt_key

            if self.embedding_key == 'query_cls':
                x_embed_mean = cls_features  # torch.Size([32, 128]) for shrec

            prompt_norm = self.l2_normalize(prompt_key, dim=1)  # torch.Size([10, 128]) for shrec
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1)  # torch.Size([32, 128]) for shrec

            # torch.Size([BS, length of prompt])
            similarity = torch.matmul(x_embed_norm, prompt_norm.t())
            B, _ = similarity.size()

            if self.decouple_prompting:
                # L2P style decoupled prompting
                _, idx = torch.topk(similarity, k=self.top_k, dim=1)
                prompt_id, id_counts = torch.unique(idx, return_counts=True)
                self.pool_dict[prompt_id] += id_counts.to(device)
                # batched_prompt_raw = prompt[idx]  # B, top_k, length, C

                # 1223 参考coda-prompt，使用enisum计算prompt
                tau = 5
                weight = F.softmax(similarity / tau, dim=1)
                batched_prompt_raw = torch.einsum('bm,mld->bld', weight, prompt)

            elif not self.decouple_prompting:
                Batch_size = similarity.size(0)
                use_similarity = similarity
                selected_keys = torch.zeros(similarity.size()).to(device)

                if self.force_select_new and train_mode == 1 and cur_task > 0 and self.random_k > 0:
                    new_idx = torch.arange(self.pool_size)[-self.random_k:].to(device)
                    new_idx = torch.broadcast_to(new_idx, (B, self.random_k))
                    _, idx = torch.topk(use_similarity[:, :-self.random_k], k=self.top_k - self.random_k, dim=1, sorted=True)
                    idx = torch.cat((idx, new_idx), 1)
                else:
                    if self.sort:
                        _, idx = torch.topk(use_similarity, k=self.top_k, dim=1, sorted=True)
                    else:
                        _, idx = torch.topk(use_similarity, k=self.top_k, dim=1, sorted=False)

                # add: 训练时强制选择对应task的prompt
                # task-specific prompt 限定
                if train_mode == 1 and self.prompt_type == 1206 and self.force_select_new:
                    idx[:] = cur_task - 1
                # 统计每个prompt被选中的次数
                prompt_id, id_counts = torch.unique(idx, return_counts=True)
                if train_mode != 1:
                    print(f'prompt_id: {prompt_id}, id_counts: {id_counts}')
                self.pool_dict[prompt_id] += id_counts.to(device)

                for b in range(Batch_size):
                    selected_keys[b, idx[b]] = 1.0

                # 不保留 selected_keys 的梯度
                selected_keys = (selected_keys - use_similarity).detach() + use_similarity
                # 保留selected_keys选择的prompt
                batched_prompt_ = selected_keys.unsqueeze(2).unsqueeze(3) * prompt.unsqueeze(
                    0)  # torch.Size([32, 16, 22, 128]) for SHREC ? 32 8 22 128
                batched_prompt_raw = torch.zeros((Batch_size, self.top_k, self.length, self.embed_dim_prompt)).to(
                    device)

                for b in range(Batch_size):
                    batched_prompt_raw[b, :, :, :] = batched_prompt_[b, idx[b], :, :]

            if self.prompt_type in [7, 17, 100, 200, 300, 27]:
                # batched_prompt_raw (N*M, T, V, Base_channels)
                # L1 output size:(N*M, Base_channels, T, V)
                # 32 128 8 22
                batched_prompt = batched_prompt_raw.permute(0, 3, 1, 2)

            elif self.prompt_type == 24:
                # mean along top k as length already is req temporal sequence.
                batched_prompt = torch.mean(batched_prompt_raw, 1)

            elif self.prompt_type == 1206:
                # 32 prompt_len 128
                batched_prompt = batched_prompt_raw
            else:
                batched_prompt = batched_prompt_raw

            out['prompt_idx'] = idx

            # Debugging, return sim as well
            out['prompt_norm'] = prompt_norm
            out['x_embed_norm'] = x_embed_norm
            out['similarity'] = similarity

            # Put pull_constraint loss calculation inside
            batched_key_norm = prompt_norm[idx]  # B, top_k, C
            out['selected_key'] = batched_key_norm
            x_embed_norm = x_embed_norm.unsqueeze(1)  # B, 1, C
            sim = batched_key_norm * x_embed_norm  # B, top_k, C
            reduce_sim = torch.sum(sim) / x_embed.shape[0]  # Scalar
            out['reduce_sim'] = reduce_sim

        else:
            '''
            Attach prompt of final shape w/o any selection 
            '''
            batched_prompt = self.prompt.unsqueeze(0)  # for batch dimension
            batched_prompt = batched_prompt.permute(0, 3, 1, 2)  # to match x_embed shape (Batch, Channel_dim, T, V)

            out['prompt_idx'] = None
            out['prompt_norm'] = None
            out['x_embed_norm'] = None
            out['similarity'] = None
            out['selected_key'] = None
            out['reduce_sim'] = None

        out['total_prompt_len'] = batched_prompt.shape[1]

        if self.prompt_type in [7, 24]:
            # Proposed solution - simply ADD
            # 32 8 22 128
            # batched_prompt = batched_prompt.permute(0, 2, 3, 1)
            # 32 176 128
            batched_prompt = batched_prompt_raw
            batched_prompt = batched_prompt.view(-1, batched_prompt.shape[1] * batched_prompt.shape[2], self.embed_dim)
            out['prompted_embedding'] = batched_prompt + x_embed

        elif self.prompt_type == 27:
            # single prompt frame, broadcast along time dim
            batched_prompt = torch.broadcast_to(batched_prompt, x_embed.size())
            out['prompted_embedding'] = batched_prompt + x_embed

        elif self.prompt_type == 100:
            # concat along feature dimension (base channel)
            batched_prompt = batched_prompt.permute(0, 2, 3, 1)
            # 32 176 128
            batched_prompt = batched_prompt.view(-1, batched_prompt.shape[1] * batched_prompt.shape[2], self.embed_dim)
            x_concat = torch.cat((x_embed, batched_prompt), dim=1)
            x_concat_t = x_concat.transpose(1, 2)
            # 降维
            out['prompted_embedding'] = self.concat_fc(x_concat_t).transpose(1, 2)

        elif self.prompt_type == 200:
            # concat along time dimension
            out['prompted_embedding'] = torch.cat((x_embed, batched_prompt), dim=2)

        elif self.prompt_type == 300:
            '''
            Apply Cross-attention between prompt & input embed
            attn_output, attn_output_weights = multihead_attn(query, key, value)

            input: (N*M, BC, T, 25) : torch.Size([128, 64, 64, 25])
            required: (N*M, T*25, BC), T*25 is the sequence length
            '''
            B, BC, T, V = batched_prompt.size()
            batched_prompt = batched_prompt.view(B, BC, -1).permute(0, 2, 1)
            x_embed = x_embed.view(B, BC, -1).permute(0, 2, 1)

            attn_output, _ = self.multihead_attn(x_embed, batched_prompt, batched_prompt)
            # out['prompted_embedding'] = attn_output.permute(0, 2, 1).view(B, BC, T, V)
            out['prompted_embedding'] = attn_output

        elif self.prompt_type == 350:
            '''
            Use cross attention operator instead of addition - POET's selection 
            '''
            batch_size, BC, T, V = x_embed.size()
            batched_prompt = torch.broadcast_to(batched_prompt, (batch_size, BC, T, V))

            batched_prompt = batched_prompt.permute(0, 2, 3, 1).reshape(batch_size, T, -1)
            x_embed = x_embed.permute(0, 2, 3, 1).reshape(batch_size, T,
                                                          -1)  # x_embed is the query; prompts are the key/value

            attn_output, _ = self.multihead_attn(x_embed, batched_prompt, batched_prompt)
            out['prompted_embedding'] = attn_output.view(batch_size, T, BC, V).permute(0, 2, 1, 3)

        elif self.prompt_type == 400:
            '''
            Use cross attention operator instead of addition - no selection and addition. cross attention without selection. 
            attention to be applied for temporal frame permutation and ordering 
            input: (N*M, BC, T, 25) : torch.Size([128, 64, 64, 25])
            required: (N*M, T*25, BC), T*25 is the sequence length
            '''
            batch_size, BC, T, V = x_embed.size()
            batched_prompt = torch.broadcast_to(batched_prompt, (batch_size, BC, T, V))

            batched_prompt = batched_prompt.permute(0, 2, 3, 1).reshape(batch_size, T, -1)
            x_embed = x_embed.permute(0, 2, 3, 1).reshape(batch_size, T,
                                                          -1)  # x_embed is the query; prompts are the key/value

            attn_output, _ = self.multihead_attn(x_embed, batched_prompt, batched_prompt)
            out['prompted_embedding'] = attn_output.view(batch_size, T, BC, V).permute(0, 2, 1, 3)

        elif self.prompt_type == 1206:
            # prefix prompt tuning

            # batch_size top_k prompt_len prompt_dim
            p_num = batched_prompt.shape[1]
            p_len = batched_prompt.shape[2]
            T = int(p_len / 2)
            prompt_s = batched_prompt[:, :, :T, :].reshape(-1, p_num * T, self.embed_dim)
            prompt_t = batched_prompt[:, :, T:, :].reshape(-1, p_num * T, self.embed_dim)

            out['prompted_embedding'] = x_embed
            out['spatial_prompt'] = prompt_s
            out['temporal_prompt'] = prompt_t

        out['selected_prompts_dict'] = self.pool_dict

        return out
