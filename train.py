import os
import torch

from data import train_dataloader_event
from utils import Adder, Timer, check_lr
from torch.utils.tensorboard import SummaryWriter
from valid import valid_event
import torch.nn.functional as F


def train_event_Temporal(model, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)


    dataloader = train_dataloader_event(args.data_dir, args.batch_size, args.num_worker)


    max_iter = len(dataloader)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, args.gamma)
    epoch = 1
    if args.resume:
        state = torch.load(args.resume)
        epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        model.load_state_dict(state['model'])
        print('Resume from %d'%epoch)
        epoch += 1

    # writer = SummaryWriter()
    epoch_pixel_adder = Adder()
    epoch_fft_adder = Adder()
    iter_pixel_adder = Adder()
    iter_fft_adder = Adder()
    epoch_timer = Timer('m')
    iter_timer = Timer('m')
    best_psnr=-1

    for epoch_idx in range(epoch, args.num_epoch + 1):
        epoch_timer.tic()
        iter_timer.tic()
        output_last_feature = None
        for iter_idx, batch_data in enumerate(dataloader):
            if iter_idx < 525:
                input_img, label_img = batch_data

                optimizer.zero_grad()
                pred_img,event_feature = model(input_img,output_last_feature) 
        
                l3 = criterion(pred_img, label_img)
                loss_content = l3

                label_fft3 = torch.rfft(label_img, signal_ndim=2, normalized=False, onesided=False)
                pred_fft3 = torch.rfft(pred_img, signal_ndim=2, normalized=False, onesided=False)

                f3 = criterion(pred_fft3, label_fft3)
                loss_fft = f3

                loss = loss_content + 0.1 * loss_fft
                loss.backward()
                optimizer.step()

                iter_pixel_adder(loss_content.item())
                iter_fft_adder(loss_fft.item())

                epoch_pixel_adder(loss_content.item())
                epoch_fft_adder(loss_fft.item())

                if (iter_idx + 1) % args.print_freq == 0:
                    lr = check_lr(optimizer)
                    print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss content: %7.4f Loss fft: %7.4f" % (
                        iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter, lr, iter_pixel_adder.average(),
                        iter_fft_adder.average()))

                    iter_timer.tic()
                    iter_pixel_adder.reset()
                    iter_fft_adder.reset()
                    
                output_last_feature = event_feature.detach()

        overwrite_name = os.path.join(args.model_save_dir, 'model.pkl')
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch_idx}, overwrite_name)

        if epoch_idx % args.save_freq == 0:
            save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % epoch_idx)
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch_idx}, save_name)
        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f" % (
            epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average(), epoch_fft_adder.average()))
        epoch_fft_adder.reset()
        epoch_pixel_adder.reset()
        scheduler.step()
        if epoch_idx % args.valid_freq == 0:
            val_gopro = valid_event(model, args)
            print('%03d epoch \n Average GOPRO PSNR %.2f dB' % (epoch_idx, val_gopro))
            if val_gopro >= best_psnr:
                torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'Best.pkl'))
    save_name = os.path.join(args.model_save_dir, 'Final.pkl')
    torch.save({'model': model.state_dict()}, save_name)

