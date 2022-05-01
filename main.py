import os
import torch
import argparse
from torch.backends import cudnn
from models.Network import build_net
from eval import eval_event
from valid import valid_event
from train import train_event_Temporal

def main(args):
    # CUDNN
    cudnn.benchmark = True

    if not os.path.exists('results/'):
        os.makedirs(args.model_save_dir)
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = build_net(args.model_name,args.base_channel,num_res=args.num_res,beta=args.beta)
    if torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.cuda()
        model = model.to(device)

    elif args.mode == 'valid':
        state = torch.load(args.valid_pkl)
        model.load_state_dict(state['model'])
        valid_event(model,args)
    elif args.mode == 'train_event_Temporal':
        train_event_Temporal(model,args)

    elif args.mode == 'test':
        state = torch.load(args.valid_pkl,map_location='cpu')

        model.load_state_dict(state['model'])
        eval_event(model,args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='STRA', choices=[
        'STRA',], type=str)
    parser.add_argument('--data_dir', type=str, default='/gdata1/caocz/Deblur/GOPRO')
    parser.add_argument('--dataset_name', type=str, default='GOPRO')
    parser.add_argument('--mode', default='train_event_Temporal', choices=['test','train_event_Temporal','valid'], type=str)
    
    parser.add_argument('--pic_size', default=64, type=int)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--event_beta', type=int, default=1e-2)
    parser.add_argument('--base_channel', type=int, default=32)
    parser.add_argument('--num_res', type=int, default=4)
    parser.add_argument('--beta', type=float, default=1e-3)
    # Train
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=50)
    parser.add_argument('--valid_freq', type=int, default=10)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--lr_steps', type=list, default=[(x+1) * 500 for x in range(3000//500)])
    # Save
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])
    # Test
    parser.add_argument('--test_model', type=str, default='')
    parser.add_argument('--model_save_dir', type=str, default='')
    parser.add_argument('--valid_pkl', type=str, default='')
    parser.add_argument('--result_dir', type=str, default='')

    args = parser.parse_args()
   
    if args.dataset_name == 'GOPRO':
        args.data_dir = '/gdata1/caocz/Deblur/GOPRO'
    elif args.dataset_name == 'HQF':
        args.data_dir = '/gdata1/caocz/Deblur/HQF'
    print(args)
    main(args)
