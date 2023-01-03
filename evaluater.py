import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import my_collate_fn, index2geohash

def evaluate(model, testset, args):
    model.eval()
    num = args.geo_character_num
    dataloader = DataLoader(testset, batch_size=args.test_batch_size, collate_fn = lambda e: my_collate_fn(e, num=num))
    batch_iterator = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, desc="evaluate")
    
    K = [5, 10, 50]
    hit_rate = [0 for _ in range(num+1+len(K))]
    for _, (src, true) in batch_iterator:
        src = src.to(args.dev)
        true = true.to(args.dev)
        tgt = torch.ones([args.test_batch_size, 1]).long().to(args.dev)
        for i in range(num+1):
            out = model(src, tgt, True)
            predict = model.predictor(out[:, -1])
            y = torch.argmax(predict, dim=1)
            tgt = torch.concat([tgt, y.unsqueeze(0).view(-1,1)], dim=1)
           
        ground_truth = true.data[:, 1:].squeeze().tolist()
        predict_result = tgt.data[:, 1:].squeeze().tolist()
   
        for i in range(num+1):
            if ground_truth[i] == predict_result[i]:
                hit_rate[i] += 1
            if i == num:
                for j in range(len(K)):
                    _, top_k = torch.topk(predict, k=K[j], dim=-1)
                    top_k = top_k.squeeze().tolist()
                    if ground_truth[i] in top_k:
                        hit_rate[i+1+j] += 1
    print('Acc of regions && HR@1,5,10,50:\n\t', [ hit_rate[i] / len(dataloader) for i in range(len(hit_rate))])    
    return [ hit_rate[i] / len(dataloader) for i in range(len(hit_rate))]

def evaluate_with_region(model, testset, region_dict:dict, args): 
    model.eval()
    num = args.geo_character_num
    dataloader = DataLoader(testset, batch_size=args.test_batch_size, collate_fn=lambda e:my_collate_fn(e, num=num))
    batch_iterator = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, desc='evaluate_with_multi_loss')
    
    hit_rate = [0 for _ in range(num+1)]
    for _, (src, true) in batch_iterator:
        src = src.to(args.dev)
        true = true.to(args.dev)
        tgt = torch.ones([args.test_batch_size, 1]).long().to(args.dev)
        for i in range(num):
            out = model(src, tgt, True)
            predict = model.predictor(out[:, -1])
            y = torch.argmax(predict[:, 2:34], dim=1) + 2
            tgt = torch.concat([tgt, y.unsqueeze(0).view(-1,1)], dim=1)
        
        out = model(src, tgt, True)
        predict = model.predictor(out[:, -1])
        region_key = index2geohash(tgt[:, 1:].tolist())[0]   
        loc_list = region_dict.get(region_key, None)
        if loc_list == None:
            predict_loc = torch.argmax(predict[:, 34:], dim=1) + 34
        else:
            predict_loc = loc_list[torch.argmax(predict[:, loc_list], dim=1).item()]
                 
        ground_truth = true.data[:, 1:].squeeze().tolist()
        predict_result = tgt.data[:, 1:].squeeze().tolist()
        
        for i in range(num):
          if ground_truth[i] == predict_result[i]:
            hit_rate[i] += 1
        if ground_truth[num] == predict_loc:
            hit_rate[num] += 1
    print('Acc of regions and loc_id:\n\t',[ hit_rate[i] / len(dataloader) for i in range(num+1)])
    return [ hit_rate[i] / len(dataloader) for i in range(num+1)]
    
def predict_with_beam_search(model, testset, region_dict:dict, args): 
    model.eval()
    num = args.geo_character_num
    dataloader = DataLoader(testset, batch_size=args.test_batch_size, collate_fn=lambda e:my_collate_fn(e,num=num)) 
    batch_iterator = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, desc='beam search')

    K = [1, 5, 10, 50]      
    hit = [0 for _ in range(len(K))]

    for _, (src, true) in batch_iterator:
        
        src = src.to(args.dev)
        true = true.to(args.dev)
        decoder_inputs = torch.ones([args.test_batch_size,1]).long().to(args.dev)
        
        beam = [(0.0, decoder_inputs[0])]
        for i in range(0, num):
            new_beam = []
            
            for score, decoder_input in beam:
                
                output = model(src, decoder_input.unsqueeze(0), True)
                predict = model.predictor(output[:, -1])
                geohash_output = torch.nn.LogSoftmax(dim=-1)(predict[:, 2:34])
                
                #geohash_score, geohash_beam = torch.topk(geohash_output, k=beam_size, dim=-1)
                geohash_output += score
                #geohash_beam += 3
                
                new_decoder_inputs = []
                for j in range(0, 32):
                    #new_decoder_input = torch.cat([decoder_input, geohash_beam[0][j].unsqueeze(0)])
                    new_decoder_input = torch.cat([decoder_input, torch.tensor([j+2]).long().cuda() ])
                    new_decoder_inputs.append(new_decoder_input)
                #new_beam.extend(list(zip(geohash_score.squeeze().tolist(), new_decoder_inputs)))
                new_beam.extend(list(zip(geohash_output.squeeze().tolist(), new_decoder_inputs)))
            beam = sorted(new_beam, key=lambda e:e[0], reverse=True)[0:args.beam_size]
            
        return_list = []
        
        for score, decoder_input in beam:
            region_key = ''
            for geohash_char in decoder_input[1:]:
                region_key += testset.idx2geo[geohash_char.item()]
            
            output = model(src, decoder_input.unsqueeze(0), True)
            predict = model.predictor(output[:, -1])
            
            #print(region_key)
            loc_list = region_dict.get(region_key, None)
            if loc_list == None:
                loc_output = torch.nn.LogSoftmax(dim=-1)(predict[:, 34:])
                loc_score_beam, loc_idx_beam = torch.topk(loc_output, k=args.beam_size, dim=-1)
                loc_score_beam += score
                loc_idx_beam += 34
                loc_beam = list(zip(loc_score_beam.squeeze().tolist(), loc_idx_beam.squeeze().tolist()))
            else:
                loc_output = torch.nn.LogSoftmax(dim=-1)(predict[:, loc_list]) + score
                if len(loc_list) > args.beam_size:
                    loc_score_beam, loc_idx_beam = torch.topk(loc_output, k=args.beam_size, dim=-1)
                    loc_beam = [(loc_score_beam.squeeze()[i].item(), loc_list[idx]) for (i,idx) in enumerate(loc_idx_beam.squeeze().tolist())] # batch_size = 1
                elif len(loc_list) == 1:
                    loc_beam = [(score, loc_list[0])]
                else:       
                    loc_beam = [(loc_output.squeeze().tolist()[i], loc) for (i, loc) in enumerate(loc_list)]
                    
            return_list.extend(loc_beam)
                
        true_id = true[0][num+1].item()
        return_list = sorted(return_list, key=lambda e:e[0], reverse=True)
        _, sorted_return_list = zip(*return_list)
        
        for i in range(len(K)):
            if true_id in sorted_return_list[0:K[i]]:
            #if true_id in sorted_return_list:
                hit[i] += 1  
    
    hit_rate = [hit[i] / len(dataloader) for i in range(len(K))]      
    print('hit rate with beam search:\n\t', hit_rate)
    return hit_rate       
