import math
import copy
import numpy as np
import transbigdata
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import Dataset

initial_geo2idx_dict = {'<pad>': 0, 'B': 1, '0': 2, '1': 3, '2': 4, '3': 5, '4': 6, '5': 7, '6': 8, '7': 9, 
                        '8': 10, '9': 11, 'b': 12, 'c': 13, 'd': 14, 'e': 15, 'f': 16, 'g': 17, 'h': 18, 
                        'j': 19, 'k': 20, 'm': 21, 'n': 22, 'p': 23, 'q': 24, 'r': 25, 's': 26, 't': 27, 
                        'u': 28, 'v': 29, 'w': 30, 'x': 31, 'y': 32, 'z': 33}
    
initial_idx2geo_dict = {0: '<pad>', 1: 'B', 2: '0', 3: '1', 4: '2', 5: '3', 6: '4', 7: '5', 8: '6', 9: '7', 
                        10: '8', 11: '9', 12: 'b', 13: 'c', 14: 'd', 15: 'e', 16: 'f', 17: 'g', 18: 'h', 
                        19: 'j', 20: 'k', 21: 'm', 22: 'n', 23: 'p', 24: 'q', 25: 'r', 26: 's', 27: 't', 
                        28: 'u', 29: 'v', 30: 'w', 31: 'x', 32: 'y', 33: 'z'}

class LBSNDataset(Dataset):
    def __init__(self, filename, geo_character_num=5):
        
        self.geo2idx = initial_geo2idx_dict.copy()
        self.idx2geo = initial_idx2geo_dict.copy()
        self.loc2gps = {'<pad>': (0.0, 0.0)}
        self.loc2geohash = {'<pad>': ''}
        self.idx2gps = {0: (0.0, 0.0)}
        self.idx2geohash = {0: ''}
        self.region2loc = {'': [] }
        self.region2idx = {'':[] }
        self.loc2count = {}
        self.n_geo = 34
        self.build_vocab_for_loc(filename, geo_character_num)
    
        self.n_user = 1
        self.n_time = 169
        self.user2idx = {'<pad>': 0}
        self.idx2user = {0: '<pad>'}
        self.user_seq = []
        self.processing(filename)
    
    def statics(self):
        print('the number of users: ', self.n_user)
        print('the number of locs: ', self.n_geo - 34)
        print('the number of regions: ', len(self.region2loc))
        total = 0
        for seq in self.user_seq:
            total += len(seq)
        print('the average length of user seq:', total / len(self.user_seq))
      
    def build_vocab_for_loc(self, filename, geo_character_num, min_freq=10):
        for line in tqdm(open(filename).readlines(), desc='build vocabulary for location', leave=False):
            line = line.strip().split('\t')
            loc = line[4]
            coordinate = (line[2], line[3]) 
            self.add_location(loc, coordinate)
        if min_freq > 0:
            self.n_geo = 34
            self.geo2idx = initial_geo2idx_dict.copy()
            self.idx2geo = initial_idx2geo_dict.copy()
            self.idx2gps = {0: (0.0, 0.0)}
            for loc in tqdm(self.loc2count, desc='filter the locs', leave=False):
                if self.loc2count[loc] >= min_freq:
                    self.add_location(loc, self.loc2gps[loc], geo_character_num, filtered=True)
        #self.locidx2freq = np.zeros(self.n_loc-1, dtype=np.int32)
        #for idx, loc in self.idx2loc.items():
        #    if idx != 0:
        #        self.locidx2freq[idx - 1] = self.loc2count[loc]
    
    def add_location(self, loc, coordinate, geo_character_num=5, filtered=False):
        if not filtered:
            if loc not in self.geo2idx:
                self.geo2idx[loc] = self.n_geo
                self.loc2gps[loc] = coordinate
                self.idx2geo[self.n_geo] = loc
                self.idx2gps[self.n_geo] = coordinate
                if loc not in self.loc2count:
                    self.loc2count[loc] = 1
                self.n_geo += 1
            else:
                self.loc2count[loc] += 1    
        else:
            if loc not in self.geo2idx:
                self.geo2idx[loc] = self.n_geo
                self.loc2gps[loc] = coordinate
                self.idx2geo[self.n_geo] = loc
                self.idx2gps[self.n_geo] = coordinate
                 
                geohash = transbigdata.geohash_encode(lat=[float(coordinate[0])], lon=[float(coordinate[1])], precision=12).loc[0]
                self.loc2geohash[loc] = geohash
                self.idx2geohash[self.n_geo] = geohash
                region = geohash[0: geo_character_num]
                if region not in self.region2loc:
                    self.region2loc[region] = list()
                    self.region2loc[region].append(loc)
                    self.region2idx[region] = list()
                    self.region2idx[region].append(self.n_geo)
                else:
                    if loc not in self.region2loc[region]:
                        self.region2loc[region].append(loc)
                        self.region2idx[region].append(self.n_geo)
                self.n_geo += 1
        
    def processing(self, filename, min_freq=20):
        temp_user_seq = {}
        for line in tqdm(open(filename).readlines(), desc='process the input file', leave=False):
            line = line.strip().split('\t')
            if len(line) < 5:
                continue
            user, time, _, _, loc = line
            if loc not in self.geo2idx:
                continue
            time = datetime.strptime(time, '%Y-%m-%dT%H:%M:%SZ')
            time_idx = time.weekday() * 24 + time.hour + 1
            loc_idx = self.geo2idx[loc]
            if user not in temp_user_seq:
                temp_user_seq[user] = list()
            temp_user_seq[user].append([loc_idx, time_idx, self.loc2geohash[loc], time])
        
        for user, seq in tqdm(temp_user_seq.items(), desc='filtering users', leave=False):
            if len(seq) >= min_freq:
                user_idx = self.n_user
                new_seq = list()
                temp_set = set()
                count = 0
                for loc_idx, t_idx, geohash, _ in sorted(seq, key=lambda e:e[3]):
                    if loc_idx in temp_set:
                        # True represents this loc has been visited by the user before 
                        new_seq.append((user_idx, loc_idx, t_idx, geohash, True))
                    else:
                        # False represents this loc is the first time to be visited by the user 
                        new_seq.append((user_idx, loc_idx, t_idx, geohash, False))
                        temp_set.add(loc_idx)
                        count += 1
                if count > min_freq / 2:
                    self.user2idx[user] = self.n_user    
                    self.idx2user[self.n_user] = user
                    self.user_seq.append(new_seq)
                    self.n_user += 1
    
    def split_slide_window(self, max_len=50):
        train_ = copy.copy(self)
        test_ = copy.copy(self)
        train_seq = list()
        test_seq = list()
        for u in range(len(self)):
            seq = self[u]
            i = 0 
            for i in reversed(range(len(seq))):
                if not seq[i][4]:
                    break
                
            if i - max_len <= 0:
                continue
            test_start = i - max_len
            test_seq.append((seq[test_start:i], seq[i:i+1]))
            for j in range(0, test_start - max_len):
                j_end = j + max_len
                if j_end < test_start:
                    train_seq.append((seq[j:j_end], seq[j_end: j_end+1]))
                else:
                    train_seq.append((seq[j:test_start-1], seq[test_start-1: test_start]))
                    
        train_.user_seq = train_seq
        test_.user_seq = sorted(test_seq, key=lambda e : len(e[0]))
        return train_, test_
        
    def split_no_overlap(self, max_len=50):
        train_ = copy.copy(self)
        test_ = copy.copy(self)
        train_seq = list()
        test_seq = list()

        for u in range(len(self)):
            seq = self[u]
            i = 0

            for i in reversed(range(len(seq))):
                if not seq[i][4]:
                    break
            
            for b in range(math.floor( (i + 1) // (max_len + 1) )):
                if (i - b * (max_len + 1)) > max_len: 
                    src = seq[b * max_len : (b+1)*max_len]
                    tgt = seq[(b+1) * max_len: (b+1) * max_len + 1]
                    train_seq.append((src, tgt))
                else:   
                    tgt = seq[i - max_len - 1 : i - max_len ]
                    src = seq[b * max_len : i - max_len - 1]
                    if len(src) == 0:
                        continue
                    if len(src) > max_len:
                        src = src[ len(src) - 1 - max_len : len(src)-1]
                    train_seq.append((src, tgt))
            test_seq.append((seq[max(0, -max_len+i):i], seq[i:i+1]))       
        train_.user_seq = train_seq
        test_.user_seq = sorted(test_seq, key=lambda e : len(e[0]))

        return train_, test_
    
    def split_tgt_seq(self, max_len=50):
        train_ = copy.copy(self)
        test_ = copy.copy(self)
        train_seq = list()
        test_seq = list()
        for u in range(len(self)):
            seq = self[u] 
            i = 0
            # i is the index of the last loc that user u first visit in the sequence 
            for i in reversed(range(len(seq))): 
                if not seq[i][4]:
                    break
            for b in range(math.floor((i + max_len - 1) // max_len)):
                if (i - b * max_len) > max_len*1.1:
                    trg = seq[(i - (b + 1) * max_len): (i - b * max_len)]
                    src = seq[(i - (b + 1) * max_len - 1): (i - b * max_len - 1)]
                    train_seq.append((src, trg))
                else:
                    trg = seq[1: (i - b * max_len)]
                    src = seq[0: (i - b * max_len - 1)]
                    train_seq.append((src, trg))
                    break
            test_seq.append((seq[max(0, -max_len+i):i], seq[i:i+1]))
        train_.user_seq = train_seq
        test_.user_seq = sorted(test_seq, key=lambda e: len(e[0]))
        return train_, test_
    
    def __len__(self):
        return len(self.user_seq) 
    
    def __getitem__(self, index):
        return self.user_seq[index]     