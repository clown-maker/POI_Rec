import os
import time
import torch
import argparse
from pathlib import Path
import _pickle as cPickle

from dataset import LBSNDataset
from model import LocPredictor
from trainer import *
from evaluater import *
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--exp', default='exp', type=str)
parser.add_argument('--dataset', default='Gowalla', type=str)
parser.add_argument('--split_method', default='no_overlap', type=str)
parser.add_argument('--max_len', default=50, type=int)
parser.add_argument('--model_dim', default=32, type=int)
parser.add_argument('--ffd_dim', default=64, type=int)
parser.add_argument('--geo_character_num', default=5, type=int)
parser.add_argument('--dev', default='cuda', type=str)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--head_num', default=2, type=int)
parser.add_argument('--num_encoder_layers', default=2, type=int)
parser.add_argument('--num_decoder_layers', default=2, type=int)
parser.add_argument('--learned_pos_embedding', action='store_true')
parser.add_argument('--use_tgt_pos', action='store_true')
parser.add_argument('--use_tgt_seg', action='store_true')

parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--weight_decay', default=0.0001, type=float)
parser.add_argument('--early_stopping', action='store_true')
parser.add_argument('--early_save_dir', default=None)
parser.add_argument('--test_batch_size', default=1, type=int)
parser.add_argument('--beam_size', default=10, type=int)

parser.add_argument('--inference_only', action='store_true')
parser.add_argument('--load_path', default=None, type=str)
parser.add_argument('--use_multi_loss', action='store_true')
parser.add_argument('--with_beam_search', action='store_true')
parser.add_argument('--eval_with_region', action='store_true')

args = parser.parse_args()
log_dir =  Path('Log') / (args.exp + '_' + args.dataset + '_') # + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
args.early_save_dir = log_dir
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
with open(log_dir / 'args.txt', 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    set_seed(87)
    
    assert args.dataset in ['Gowalla', 'Foursquare', 'Brightkite']
    dataset_path = Path('data') / Path('dataset') / (args.dataset + '_dataset_num_' + str(args.geo_character_num) + '.pickle')
    if os.path.isfile(dataset_path):
        with open(dataset_path, 'rb') as file:
            dataset = cPickle.load(file)
    else:
        rawfile_path = Path('data') / Path('rawfile') / (args.dataset + '.txt') 
        dataset = LBSNDataset(rawfile_path, args.geo_character_num)
        with open(dataset_path, 'wb') as file:
            cPickle.dump(dataset, file)
    dataset.statics()
    
    assert args.split_method in ['slide_window', 'no_overlap', 'tgt_seq'], 'Invalid Split Method!'
    if args.split_method == 'slide_window':
        trainset, testset = dataset.split_slide_window(args.max_len)
    elif args.split_method == 'no_overlap':
        trainset, testset = dataset.split_no_overlap(args.max_len)
    elif args.split_method == 'tgt_seq':
        trainset, testset = dataset.split_tgt_seq(args.max_len)
    print('the length of trainset: ', len(trainset))
    print('the length of testset: ', len(testset))
        
    model = LocPredictor(dataset.n_user, dataset.n_geo, dataset.n_time, args).to(args.dev)
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            print('failed init ' + name)
            pass # just ignore those failed init layers
    
    if args.inference_only:
        load_path = Path('log') / args.load_path
        model.load(load_path)
    else:
        if args.use_multi_loss:
            train_loss_list, test_loss_list, region_loss_list, loc_loss_list= train_with_multiLoss(model, trainset, testset, args)
        else:
            train_loss_list, test_loss_list = train(model, trainset, testset, args)
        fname = 'model.epoch={}.lr={}.dim={}.maxlen={}.num={}.pkl'
        fname = fname.format(args.epochs, args.lr, args.model_dim, args.max_len, args.geo_character_num)
        model.save(log_dir / fname)
                    
    if args.with_beam_search:
        hit_rate_beam_search = predict_with_beam_search(model, testset, dataset.region2idx, args)
    
    if args.eval_with_region:
        hit_rate = evaluate_with_region(model, testset, dataset.region2idx, args)
    else:
        hit_rate = evaluate(model, testset, args)
             
    if not args.inference_only:
        figure_path = log_dir / 'figure.png'
        if args.use_multi_loss:
            draw_figure(figure_path, train_loss_list, test_loss_list, 
                        region_loss=region_loss_list, 
                        loc_loss=loc_loss_list)
        else:
            draw_figure(figure_path, train_loss_list, test_loss_list)
    
    log_path = log_dir / 'log.txt'
    with open(log_path, 'w') as f:
        f.write("hit rate:\n")
        str_hit_rate = '\t' + '\t'.join([ str(hit_rate[i]) for i in range(len(hit_rate)) ])
        f.writelines(str_hit_rate)
        if args.with_beam_search:
            f.write("\nhit rate with beam search:\n")
            str_hit_rate_beam_search = '\t' + '\t'.join([str(hit_rate_beam_search[i]) for i in range(len(hit_rate_beam_search))])
            f.writelines(str_hit_rate_beam_search)  
        
    

    print("Done")
