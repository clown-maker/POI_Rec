import math
import utils
import torch
import numpy as np
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, len):
        return self.pe[:, :len, :]        

class LocPredictor(nn.Module):
    def __init__(self, n_user, n_geo, n_time, config:dict):
        super(LocPredictor, self).__init__()
        self.config = config
        self.max_len = config.max_len
        self.geo_character_num = config.geo_character_num
        self.dev = config.dev
        
        if config.learned_pos_embedding == True:
            self.pos_emb = nn.Embedding(self.max_len+1, config.model_dim, padding_idx=0) # 1 : padding
        else:
            self.pos_emb = PositionalEmbedding(config.model_dim)
        self.seg_emb = nn.Embedding(self.geo_character_num+2, config.model_dim, padding_idx=0) # 2: other(padding), location_id
        
        self.geo_emb = nn.Embedding(n_geo, config.model_dim, padding_idx=0)
        
        self.emb_dropout = nn.Dropout(p=config.dropout)
        self.transformer = nn.Transformer(
                                          d_model=config.model_dim, 
                                          nhead=config.head_num, 
                                          num_encoder_layers=config.num_encoder_layers, 
                                          dim_feedforward=config.ffd_dim, 
                                          num_decoder_layers=config.num_decoder_layers, 
                                          batch_first=True,
                                          dropout=config.dropout
                                          )
        
        self.predictor = nn.Linear(config.model_dim, n_geo)
    
    def src_geo2emb(self, src_geo):
        src = self.geo_emb(src_geo)
        
        #src pos embedding 
        assert src_geo.size(1) % (self.config.geo_character_num + 1) == 0, \
            "The number of src geo_tokens should be a multiple of (geo_character_num+1)" 
        src_len = src_geo.size(1) // (self.config.geo_character_num + 1)
        if self.config.learned_pos_embedding:
            positions = np.tile(np.array(range(1, src_len+1)), [src_geo.size(0), 1])
            src_pos_emb = self.pos_emb(torch.LongTensor(positions).to(self.dev))
            src += src_pos_emb.repeat_interleave(self.geo_character_num+1, dim=1)
        else:
            src += self.pos_emb(src_len).repeat_interleave(self.geo_character_num+1, dim=1)
        
        # src seg embedding  
        segments = np.tile(np.array(range(1, self.geo_character_num+2)), [src_geo.size(0), src_len])
        src += self.seg_emb(torch.LongTensor(segments).to(self.dev))
        src = self.emb_dropout(src)
        return src
    
    def tgt_geo2emb(self, tgt_geo, isPredict):
        tgt = self.geo_emb(tgt_geo) 
        if not isPredict:
            assert (tgt_geo.size(1) + 1) % (self.config.geo_character_num + 1) == 1, \
                    "The number of tgt geo_tokens should be a multiple of (geo_character_num+1) with a remainder of 1"
            tgt_len = (tgt_geo.size(1) + 1) // (self.config.geo_character_num + 1) 
            if self.config.use_tgt_pos:  
                if self.config.learned_pos_embedding:
                    pad_pos_emb = self.pos_emb(torch.zeros(tgt_geo.size(0), 1).long().to(self.dev))
                    positions = np.tile(np.array(range(1, tgt_len+1)), [tgt_geo.size(0), 1])
                    tgt_pos_emb = self.pos_emb(torch.LongTensor(positions).to(self.dev)).repeat_interleave(self.geo_character_num+1, dim=1)[:, :-1, :]
                    tgt += torch.concat([pad_pos_emb, tgt_pos_emb], dim=1)
                else:
                    pad_pos_emb = self.pos_emb(1)
                    tgt_pos_emb = self.pos_emb(tgt_len).to(self.dev).repeat_interleave(self.geo_character_num+1, dim=1)[:, :-1, :] 
                    tgt += torch.concat([pad_pos_emb, tgt_pos_emb], dim=1)  
            if self.config.use_tgt_seg:   
                pad_seg_emb = self.seg_emb(torch.zeros(tgt_geo.size(0), 1).long().to(self.dev))
                segments = np.tile(np.array(range(1, self.geo_character_num+2)), [tgt_geo.size(0), tgt_len])
                tgt_seg_emb = self.seg_emb(torch.LongTensor(segments).to(self.dev))[:, :-1, :]
                tgt += torch.concat([pad_seg_emb, tgt_seg_emb], dim=1)     
        else:
            if self.config.use_tgt_pos: 
                if self.config.learned_pos_embedding:
                    positions = np.tile(np.array([0] + [1] * (tgt_geo.size(1)-1)), [tgt_geo.size(0), 1])
                    tgt_pos_emb = self.pos_emb(torch.LongTensor(positions).to(self.dev))
                    tgt += tgt_pos_emb
                else:
                    tgt += self.pos_emb(1).repeat_interleave(tgt_geo.size(1), dim=1)
            if self.config.use_tgt_seg:
                segments = np.tile(np.array(range(tgt_geo.size(1))), [tgt_geo.size(0), 1])
                tgt_seg_emb = self.seg_emb(torch.LongTensor(segments).to(self.dev))
                tgt += tgt_seg_emb
        tgt = self.emb_dropout(tgt)  
        return tgt
    
    def forward(self, src_geo, tgt_geo, isPredict=False):
        src_padding_mask = utils.generate_padding_mask(src_geo).to(self.dev)
        tgt_padding_mask = utils.generate_padding_mask(tgt_geo).to(self.dev)
        tgt_mask = utils.generate_square_subsequent_mask(tgt_geo.size()[-1], device=self.dev)
        
        src = self.src_geo2emb(src_geo)
        tgt = self.tgt_geo2emb(tgt_geo, isPredict)     
        
        # 后续需要将encoder和decoder解耦
        out = self.transformer(src, tgt,
                               tgt_mask=tgt_mask,
                               src_key_padding_mask=src_padding_mask,
                               tgt_key_padding_mask=tgt_padding_mask)
        return out
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))