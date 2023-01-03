import numpy as np
from matplotlib import pyplot as plt

def draw_loss_figure(train_loss_list, test_loss_list, path, region_loss_list=None, loc_loss_list=None, **kwags):
    assert len(test_loss_list) == len(train_loss_list)
    legend_list = ['train', 'test']
    plt.plot(range(len(train_loss_list)), train_loss_list, '.-', label='train' )
    plt.plot(range(len(test_loss_list)), test_loss_list, '.-', label='test')
    plt.xlabel('epoches', fontsize=20)
    plt.ylabel('loss', fontsize=20)
    if region_loss_list != None:
        assert len(test_loss_list) == len(region_loss_list)
        legend_list.append('region')
        plt.plot(range(len(region_loss_list)), region_loss_list, '.-', label='region')
    if loc_loss_list != None:
        assert len(loc_loss_list) == len(test_loss_list)
        legend_list.append('loc')
        plt.plot(range(len(loc_loss_list)), loc_loss_list, '.-', label='loc')
    for key, value in kwags.items():
        plt.plot(range(len(value)), value, '.-', label=str(key))
        legend_list.append(str(key))
    plt.legend()
    plt.savefig(path)

list1 = np.arange(10)
list2 = np.arange(10) ** 2
list3 = np.arange(10) * 3
list4 = np.arange(8) + 2 
draw_loss_figure(list1, list2, 'test.png', acc1=list3, acc2=list4)
a = np.tile(np.array(0), [4, 1])
print(a)
quit()
import numpy as np
import torch 
import torch.nn as nn

emb = nn.Embedding(100, 32, padding_idx=0)
tgt_geo = torch.rand(2,6)

pad_pos = np.tile(np.array(0), [tgt_geo.size(0), 1])
pad_pos_emb = emb(torch.LongTensor(pad_pos))

#pad_pos_emb = emb(torch.zeros(tgt_geo.size(0),1).long())

tgt_len = 3
num = tgt_geo.size(1) // tgt_len
positions = np.tile(np.array(range(1, tgt_len+1)), [tgt_geo.size(0), 1])
tgt_pos_emb = emb(torch.LongTensor(positions)).repeat_interleave(num, dim=1)

tgt = torch.concat([pad_pos_emb, tgt_pos_emb], dim=1)

print(pad_pos_emb)
print(tgt_pos_emb)
print(tgt)