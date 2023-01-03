import torch 
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from early_stopping import EarlyStopping
from torch.nn.utils.rnn import pad_sequence
from utils import my_collate_fn, index2geohash, get_loc_tgt

def train(model, trainset, testset, args):
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = CrossEntropyLoss()
    train_loss_list = []
    test_loss_list = []
    
    if args.early_stopping:
        early_stopping = EarlyStopping(args.early_save_dir, patience=5) 
    
    for _ in tqdm(range(args.epochs), desc='train_epochs'):
        dataloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, collate_fn = lambda e: my_collate_fn(e, num=args.geo_character_num))
        batch_iterator = tqdm(enumerate(dataloader), total=len(dataloader), desc='train', leave=False)
        running_loss = 0.0
        processed_batch = 0
        
        model.train()
        for _, (src, tgt) in batch_iterator:
            src = src.to(args.dev)
            tgt = tgt.to(args.dev)
            
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])
            output = model.predictor(output)
            loss = loss_fn(output.contiguous().view(-1, output.size(-1)), tgt[:, 1:].contiguous().view(-1))
            loss.backward()     
            optimizer.step()       
            running_loss += loss.item()
            processed_batch += 1
        train_loss_list.append(running_loss / processed_batch)   
        
        model.eval()
        testdataloader = DataLoader(testset, batch_size=args.batch_size, collate_fn=lambda e: my_collate_fn(e, num=args.geo_character_num))
        test_batch_iterator = tqdm(enumerate(testdataloader), total=len(testdataloader), desc='test', leave=False)
        test_running_loss = 0.0
        test_batch = 0
        with torch.no_grad():
            for _, (test_src, test_tgt) in test_batch_iterator:
                test_src = test_src.to(args.dev)
                test_tgt = test_tgt.to(args.dev)
                
                test_output = model(test_src, test_tgt[:, :-1])
                test_output = model.predictor(test_output)
                test_loss = loss_fn(test_output.contiguous().view(-1, output.size(-1)), test_tgt[:, 1:].contiguous().view(-1))
                test_running_loss += test_loss.item()
                test_batch += 1
        test_loss_list.append(test_running_loss / test_batch)
        
        if args.early_stopping:
            early_stopping(test_running_loss / test_batch, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    return train_loss_list, test_loss_list

def train_with_multiLoss(model, trainset, testset, args):
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = CrossEntropyLoss()
    train_loss_list = []
    test_loss_list = []
    region_loss_list = []
    loc_loss_list = []
    
    if args.early_stopping:
        early_stopping = EarlyStopping(args.early_save_dir, patience=5) 
    
    for _ in tqdm(range(args.epochs), desc='train_epochs'):
        dataloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, collate_fn = lambda e: my_collate_fn(e, num=args.geo_character_num))
        batch_iterator = tqdm(enumerate(dataloader), total=len(dataloader), desc='train', leave=False)
        running_loss = 0.0
        processed_batch = 0
        
        model.train()
        for _, (src, tgt) in batch_iterator:
            src = src.to(args.dev)
            tgt = tgt.to(args.dev)
    
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])
            output = model.predictor(output)
            
            geohash_output = output[:, :-1, 2:34]
            geohash_tgt = tgt[:, 1:-1] - 2
            loss1 = loss_fn(geohash_output.contiguous().view(-1, geohash_output.size(-1)), geohash_tgt.contiguous().view(-1))
            
            geohash_list = index2geohash(tgt[:, 1:-1].tolist())
            loc_list = [trainset.region2idx[reg] for reg in geohash_list]
            loc_output = [output[:,-1,:][i,loc_list[i]] for i in range(len(src)) ]
            loc_output = pad_sequence(loc_output, batch_first=True, padding_value=-torch.inf)
            loc_tgt = get_loc_tgt(loc_list, tgt[:, -1].tolist()).cuda()
            loss2 = loss_fn(loc_output.contiguous().view(-1, loc_output.size(-1)), loc_tgt.contiguous().view(-1))
            
            loss = loss1 + loss2
            loss.backward()     
            optimizer.step()       
            running_loss += loss.item()
            processed_batch += 1
        train_loss_list.append(running_loss / processed_batch) 
        
        model.eval()
        testdataloader = DataLoader(testset, batch_size=args.batch_size, collate_fn=lambda e: my_collate_fn(e, num=args.geo_character_num))
        test_batch_iterator = tqdm(enumerate(testdataloader), total=len(testdataloader), desc='test', leave=False)
        test_loss = 0.0
        region_loss = 0.0
        loc_loss = 0.0
        test_batch = 0
        with torch.no_grad():
            for _, (test_src, test_tgt) in test_batch_iterator:
                test_src = test_src.to(args.dev)
                test_tgt = test_tgt.to(args.dev)
                
                test_output = model(test_src, test_tgt[:, :-1])
                test_output = model.predictor(test_output)
                
                test_geohash_output = test_output[:, :-1, 2:34]
                test_geohash_tgt = test_tgt[:, 1:-1] - 2
                test_loss1 = loss_fn(test_geohash_output.contiguous().view(-1, test_geohash_output.size(-1)), test_geohash_tgt.contiguous().view(-1))
            
                test_geohash_list = index2geohash(test_tgt[:, 1:-1].tolist())
                test_loc_list = [testset.region2idx[reg] for reg in test_geohash_list]
                test_loc_output = [test_output[:,-1,:][i, test_loc_list[i]] for i in range(len(src)) ]
                test_loc_output = pad_sequence(test_loc_output, batch_first=True, padding_value=-torch.inf)
                test_loc_tgt = get_loc_tgt(test_loc_list, test_tgt[:, -1].tolist()).cuda()
                test_loss2 = loss_fn(test_loc_output.contiguous().view(-1, test_loc_output.size(-1)), test_loc_tgt.contiguous().view(-1))
                #cur_test_loss = loss_fn(test_output.contiguous().view(-1, output.size(-1)), test_tgt[:, 1:].contiguous().view(-1))
                
                region_loss += test_loss1.item()
                loc_loss += test_loss2.item()
                test_loss += (test_loss1.item() + test_loss2.item())
                test_batch += 1
        test_loss_list.append(test_loss / test_batch)
        region_loss_list.append(region_loss / test_batch)
        loc_loss_list.append(loc_loss / test_batch)
        
        if args.early_stopping:
            early_stopping(test_loss / test_batch, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
    return train_loss_list, test_loss_list, region_loss_list, loc_loss_list