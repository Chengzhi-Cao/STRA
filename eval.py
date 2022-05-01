import os
import torch
from torchvision.transforms import functional as F
import numpy as np
from utils import Adder
from skimage.metrics import peak_signal_noise_ratio
import time


from data import valid_dataloader_event,valid_dataloader_HQF
from utils import Adder
import os
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
import numpy as np
import time

def eval_event(model, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.dataset_name == 'GOPRO':
        gopro = valid_dataloader_event(args.data_dir, batch_size=1, num_workers=0)
    elif args.dataset_name == 'HQF':
        gopro = valid_dataloader_HQF(args.data_dir, batch_size=1, num_workers=0)
    model.eval()
    psnr_adder = Adder()
    ssim_adder = Adder()
    time_list = []
    with torch.no_grad():
        print('Start GoPro Evaluation')
        for idx, data in enumerate(gopro):
            input_img, label_img = data

            ###################################
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            #######################################
            start = time.clock()
            pred,_ = model(input_img)
            end = time.clock()
            
            pred_clip = torch.clamp(pred, 0, 1)
            p_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()
            if args.save_image:
                save_name = os.path.join(args.result_dir,'{}.jpg'.format(idx))
                pred_clip += 0.5 / 255
                pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred = pred.convert('L')
                pred.save(save_name)
                print('index=',idx)

            psnr = peak_signal_noise_ratio(p_numpy, label_numpy, data_range=1)
            
            p_numpy = p_numpy.transpose(2,1,0)
            label_numpy = label_numpy.transpose(2,1,0)

            ssim = structural_similarity(p_numpy,label_numpy,multichannel=True) #[3,720,720]
            psnr_adder(psnr)
            ssim_adder(ssim)
            running_time = end - start

    return psnr_adder.average()