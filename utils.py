import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence

base32_dict = {'0': 2, '1': 3, '2': 4, '3': 5, '4': 6, '5': 7, '6': 8, '7': 9, '8': 10, 
               '9': 11, 'b': 12, 'c': 13, 'd': 14, 'e': 15, 'f': 16, 'g': 17, 'h': 18, 
               'j': 19, 'k': 20, 'm': 21, 'n': 22, 'p': 23, 'q': 24, 'r': 25, 's': 26, 
               't': 27, 'u': 28, 'v': 29, 'w': 30, 'x': 31, 'y': 32, 'z': 33}

idx2geo_dict = {2: '0', 3: '1', 4: '2', 5: '3', 6: '4', 7: '5', 8: '6', 9: '7', 10: '8', 
                11: '9', 12: 'b', 13: 'c', 14: 'd', 15: 'e', 16: 'f', 17: 'g', 18: 'h', 
                19: 'j', 20: 'k', 21: 'm', 22: 'n', 23: 'p', 24: 'q', 25: 'r', 26: 's', 
                27: 't', 28: 'u', 29: 'v', 30: 'w', 31: 'x', 32: 'y', 33: 'z'}

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def draw_figure(figure_path, train_loss_list, test_loss_list, **kwags):
    assert len(test_loss_list) == len(train_loss_list)
    legend_list = ['train_loss', 'test_loss']
    plt.plot(range(len(train_loss_list)), train_loss_list, '.-', label='train' )
    plt.plot(range(len(test_loss_list)), test_loss_list, '.-', label='test')
    plt.xlabel('epoches', fontsize=20)
    plt.ylabel('indicator', fontsize=20)
    for key, value in kwags.items():
        assert len(value) == len(train_loss_list)
        plt.plot(range(len(value)), value, '.-', label=str(key))
        legend_list.append(str(key))
    plt.legend()
    plt.savefig(figure_path)

def geohash2index(geohash, num):
    list =  [base32_dict[geohash[i]] for i in range(num)]
    return list

def index2geohash(index_lists:list)->list:
    geohash_list = []
    for idx_list in index_lists:
        geohash = ''
        for idx in idx_list:
            geohash += idx2geo_dict[idx]
        geohash_list.append(geohash)
    return geohash_list

def generate_padding_mask(inputs):
    padding_mask = torch.zeros(inputs.size())
    padding_mask[inputs == 0] = -torch.inf
    return padding_mask.bool()

def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def get_loc_tgt(loc_list:list, tgt:list):
    assert len(loc_list) == len(tgt)
    loc_tgt = []
    for i in range(len(tgt)):
        loc_tgt.append(loc_list[i].index(tgt[i]))
    return torch.tensor(loc_tgt)

def my_collate_fn(batch, num=5):
    src, tgt = [], []
    for (src_seq, tgt_seq) in batch:
        new_src_seq = list()
        for src_item in src_seq:
            for i in range(num):
                new_src_seq.append(base32_dict[src_item[3][i]])
            new_src_seq.append(src_item[1])
        src.append(torch.tensor(new_src_seq))
        
        new_tgt_seq = [1]
        for tgt_item in tgt_seq:
            for i in range(num):
                new_tgt_seq.append(base32_dict[tgt_item[3][i]])
            new_tgt_seq.append(tgt_item[1])
        tgt.append(torch.tensor(new_tgt_seq))

    src_batch = pad_sequence(src, padding_value=0, batch_first=True)
    tgt_batch = pad_sequence(tgt, padding_value=0, batch_first=True)
    
    return src_batch, tgt_batch