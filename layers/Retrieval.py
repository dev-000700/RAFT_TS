import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

class RetrievalTool():
    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
        n_period=3,
        temperature=0.1,
        topm=3,
        with_dec=False,
        return_key=False,
    ):
        period_num = [16, 8, 4, 2, 1]
        period_num = period_num[-1 * n_period:]
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        
        self.n_period = n_period
        self.period_num = sorted(period_num, reverse=True)
        
        self.temperature = temperature
        self.topm = topm
        
        self.with_dec = with_dec
        self.return_key = return_key
        
        # DEBUG: In ra thông tin khởi tạo
        print(f"[DEBUG] RetrievalTool initialized:")
        print(f"  - topm: {self.topm}")
        print(f"  - temperature: {self.temperature}")
        print(f"  - n_period: {self.n_period}")
        print(f"  - period_num: {self.period_num}")
        
    def prepare_dataset(self, train_data):
        print(f"[DEBUG] Preparing dataset with {len(train_data)} samples")
        
        train_data_all = []
        y_data_all = []

        for i in range(len(train_data)):
            td = train_data[i]
            train_data_all.append(td[1])
            
            if self.with_dec:
                y_data_all.append(td[2][-(train_data.pred_len + train_data.label_len):])
            else:
                y_data_all.append(td[2][-train_data.pred_len:])
            
        self.train_data_all = torch.tensor(np.stack(train_data_all, axis=0)).float()
        self.train_data_all_mg, _ = self.decompose_mg(self.train_data_all)
        
        self.y_data_all = torch.tensor(np.stack(y_data_all, axis=0)).float()
        self.y_data_all_mg, _ = self.decompose_mg(self.y_data_all)

        self.n_train = self.train_data_all.shape[0]
        
        # DEBUG: In thông tin dataset
        print(f"[DEBUG] Dataset prepared:")
        print(f"  - train_data_all shape: {self.train_data_all.shape}")
        print(f"  - y_data_all shape: {self.y_data_all.shape}")
        print(f"  - n_train: {self.n_train}")

    def decompose_mg(self, data_all, remove_offset=True):
        data_all = copy.deepcopy(data_all) # T, S, C

        mg = []
        for g in self.period_num:
            cur = data_all.unfold(dimension=1, size=g, step=g).mean(dim=-1)
            cur = cur.repeat_interleave(repeats=g, dim=1)
            
            mg.append(cur)
            
        mg = torch.stack(mg, dim=0) # G, T, S, C

        if remove_offset:
            offset = []
            for i, data_p in enumerate(mg):
                cur_offset = data_p[:,-1:,:]
                mg[i] = data_p - cur_offset
                offset.append(cur_offset)
        else:
            offset = None
            
        offset = torch.stack(offset, dim=0)
            
        return mg, offset
    
    def periodic_batch_corr(self, data_all, key, in_bsz = 512):
        _, bsz, features = key.shape
        _, train_len, _ = data_all.shape
        
        bx = key - torch.mean(key, dim=2, keepdim=True)
        
        iters = math.ceil(train_len / in_bsz)
        
        sim = []
        for i in range(iters):
            start_idx = i * in_bsz
            end_idx = min((i + 1) * in_bsz, train_len)
            
            cur_data = data_all[:, start_idx:end_idx].to(key.device)
            ax = cur_data - torch.mean(cur_data, dim=2, keepdim=True)
            
            cur_sim = torch.bmm(F.normalize(bx, dim=2), F.normalize(ax, dim=2).transpose(-1, -2))
            sim.append(cur_sim)
            
        sim = torch.cat(sim, dim=2)
        
        return sim
        
    def retrieve(self, x, index, train=True, debug=False):
        index = index.to(x.device)
        
        bsz, seq_len, channels = x.shape
        assert(seq_len == self.seq_len, channels == self.channels)
        
        # DEBUG: In thông tin tham số
        if debug:
            print(f"[DEBUG] Retrieve called with:")
            print(f"  - topm: {self.topm}")
            print(f"  - temperature: {self.temperature}")
            print(f"  - batch_size: {bsz}")
        
        x_mg, mg_offset = self.decompose_mg(x) # G, B, S, C

        sim = self.periodic_batch_corr(
            self.train_data_all_mg.flatten(start_dim=2), # G, T, S * C
            x_mg.flatten(start_dim=2), # G, B, S * C
        ) # G, B, T
        
        # DEBUG: Kiểm tra similarity distribution
        if debug:
            print(f"[DEBUG] Similarity stats:")
            print(f"  - sim shape: {sim.shape}")
            print(f"  - sim min: {sim.min().item():.6f}")
            print(f"  - sim max: {sim.max().item():.6f}")
            print(f"  - sim mean: {sim.mean().item():.6f}")
            print(f"  - sim std: {sim.std().item():.6f}")
            
        if train:
            sliding_index = torch.arange(2 * (self.seq_len + self.pred_len) - 1).to(x.device)
            sliding_index = sliding_index.unsqueeze(dim=0).repeat(len(index), 1)
            sliding_index = sliding_index + (index - self.seq_len - self.pred_len + 1).unsqueeze(dim=1)
            
            sliding_index = torch.where(sliding_index >= 0, sliding_index, 0)
            sliding_index = torch.where(sliding_index < self.n_train, sliding_index, self.n_train - 1)

            self_mask = torch.zeros((bsz, self.n_train)).to(x.device)
            self_mask = self_mask.scatter_(1, sliding_index, 1.)
            self_mask = self_mask.unsqueeze(dim=0).repeat(self.n_period, 1, 1)
            
            sim = sim.masked_fill_(self_mask.bool(), float('-inf')) # G, B, T
            
            # DEBUG: Kiểm tra masked similarity
            if debug:
                valid_sim = sim[sim != float('-inf')]
                print(f"[DEBUG] After masking:")
                print(f"  - valid similarities: {len(valid_sim)}")
                print(f"  - valid sim mean: {valid_sim.mean().item():.6f}")

        sim = sim.reshape(self.n_period * bsz, self.n_train) # G X B, T
                
        topm_index = torch.topk(sim, self.topm, dim=1).indices
        topm_values = torch.topk(sim, self.topm, dim=1).values
        
        # DEBUG: Kiểm tra top-k selection
        if debug:
            print(f"[DEBUG] Top-k selection:")
            print(f"  - topm_index shape: {topm_index.shape}")
            print(f"  - First sample top-{min(5, self.topm)} similarities:")
            print(f"    {topm_values[0, :min(5, self.topm)]}")
            print(f"  - Similarity gaps (diff between consecutive):")
            if self.topm > 1:
                gaps = tomp_values[0, :-1] - topm_values[0, 1:]
                print(f"    {gaps[:min(4, len(gaps))]}")
        
        ranking_sim = torch.ones_like(sim) * float('-inf')
        
        rows = torch.arange(sim.size(0)).unsqueeze(-1).to(sim.device)
        ranking_sim[rows, topm_index] = sim[rows, topm_index]
        
        sim = sim.reshape(self.n_period, bsz, self.n_train) # G, B, T
        ranking_sim = ranking_sim.reshape(self.n_period, bsz, self.n_train) # G, B, T

        data_len, seq_len, channels = self.train_data_all.shape
            
        ranking_prob = F.softmax(ranking_sim / self.temperature, dim=2)
        
        # DEBUG: Kiểm tra probability distribution
        if debug:
            print(f"[DEBUG] Probability distribution:")
            print(f"  - ranking_prob shape: {ranking_prob.shape}")
            # Lấy sample đầu tiên, period đầu tiên
            prob_sample = ranking_prob[0, 0, :]
            top_probs = prob_sample.topk(min(10, self.topm))
            print(f"  - Top-{min(10, self.topm)} probabilities:")
            print(f"    values: {top_probs.values}")
            print(f"    indices: {top_probs.indices}")
            
            # Kiểm tra entropy để xem distribution có đồng đều không
            entropy = -(prob_sample * torch.log(prob_sample + 1e-10)).sum()
            max_entropy = math.log(self.n_train)
            print(f"  - Entropy: {entropy:.4f} / {max_entropy:.4f} (higher = more uniform)")
            print(f"  - Effective number of patterns: {torch.exp(entropy):.2f}")
        
        ranking_prob = ranking_prob.detach().cpu() # G, B, T
        
        y_data_all = self.y_data_all_mg.flatten(start_dim=2) # G, T, P * C
        
        pred_from_retrieval = torch.bmm(ranking_prob, y_data_all).reshape(self.n_period, bsz, -1, channels)
        pred_from_retrieval = pred_from_retrieval.to(x.device)
        
        # DEBUG: In thông tin kết quả
        if debug:
            print(f"[DEBUG] Retrieval output:")
            print(f"  - pred_from_retrieval shape: {pred_from_retrieval.shape}")
            print(f"  - pred mean: {pred_from_retrieval.mean().item():.6f}")
            print(f"  - pred std: {pred_from_retrieval.std().item():.6f}")
        
        return pred_from_retrieval
    
    def retrieve_all(self, data, train=False, device=torch.device('cpu'), debug_first_batch=False):
        assert(self.train_data_all_mg != None)
        
        rt_loader = DataLoader(
            data,
            batch_size=1024,
            shuffle=False,
            num_workers=8,
            drop_last=False
        )
        
        retrievals = []
        with torch.no_grad():
            for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(rt_loader)):
                # DEBUG: Chỉ debug batch đầu tiên
                debug = debug_first_batch and i == 0
                
                pred_from_retrieval = self.retrieve(
                    batch_x.float().to(device), 
                    index, 
                    train=train,
                    debug=debug
                )
                pred_from_retrieval = pred_from_retrieval.cpu()
                retrievals.append(pred_from_retrieval)
                
        retrievals = torch.cat(retrievals, dim=1)
        print(f"Retrieved shape: {retrievals.shape}")  # G, B, P, C
        
        # DEBUG: Thống kê tổng thể
        if debug_first_batch:
            print(f"[DEBUG] Final retrieval stats:")
            print(f"  - Total samples processed: {retrievals.shape[1]}")
            print(f"  - Final output mean: {retrievals.mean().item():.6f}")
            print(f"  - Final output std: {retrievals.std().item():.6f}")
        
        return retrievals

# Cách sử dụng debug:
# rt = RetrievalTool(seq_len=96, pred_len=96, channels=1, topm=5, temperature=0.1)
# result = rt.retrieve_all(data, debug_first_batch=True)
