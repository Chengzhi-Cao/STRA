import os
import torch
import numpy as np
from PIL import Image as Image
from data import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import torch.utils.data as udata
import cv2
import torchvision.transforms as transforms
import random


def train_test_dataloader(path, batch_size=64, num_workers=0, use_transform=True):
    image_dir = os.path.join(path, 'valid')

    transform = None
    if use_transform:
        transform = PairCompose(
            [
                PairRandomCrop(256),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ]
        )
    dataloader = DataLoader(
        DeblurDataset(image_dir, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'blur/'))
        # self._check_image(self.image_list)
        self.image_list.sort()
        self.image_list = self.image_list[:-1]
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'blur', self.image_list[idx]))
        label = Image.open(os.path.join(self.image_dir, 'sharp', self.image_list[idx]))

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
        if self.is_test:
            name = self.image_list[idx]
            return image, label, name
        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError


#########################################################################################################
class PairRandomCrop_4(transforms.RandomCrop):

    def __call__(self, image, label,event_output_1,event_output_2):

        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)
            
            event_output_1 = F.pad(event_output_1, self.padding, self.fill, self.padding_mode)
            event_output_2 = F.pad(event_output_2, self.padding, self.fill, self.padding_mode)
        # pad the width if needed
        if self.pad_if_needed and image.size[0] < self.size[1]:
            image = F.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
            label = F.pad(label, (self.size[1] - label.size[0], 0), self.fill, self.padding_mode)
            event_output_1 = F.pad(event_output_1, (self.size[1] - event_output_1.size[0], 0), self.fill, self.padding_mode)
            event_output_2 = F.pad(event_output_2, (self.size[1] - event_output_2.size[0], 0), self.fill, self.padding_mode)
        
        # pad the height if needed
        if self.pad_if_needed and image.size[1] < self.size[0]:
            image = F.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            label = F.pad(label, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            event_output_1 = F.pad(event_output_1, (0, self.size[0] - event_output_1.size[1]), self.fill, self.padding_mode)
            event_output_2 = F.pad(event_output_2, (0, self.size[0] - event_output_2.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(image, self.size)

        return F.crop(image, i, j, h, w), F.crop(label, i, j, h, w),F.crop(event_output_1, i, j, h, w),F.crop(event_output_2, i, j, h, w)


class PairCompose_4(transforms.Compose):
    def __call__(self, image, label,event_output_1,event_output_2):
        for t in self.transforms:
            image, label,event_output_1,event_output_2 = t(image, label,event_output_1,event_output_2)
        return image, label,event_output_1,event_output_2


class PairRandomHorizontalFilp_4(transforms.RandomHorizontalFlip):
    def __call__(self, img, label,event_output_1,event_output_2):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:    # self.p=0.5
            
            return F.hflip(img), F.hflip(label),F.hflip(event_output_1),F.hflip(event_output_2)
        return img, label,event_output_1,event_output_2


class PairToTensor_4(transforms.ToTensor):
    def __call__(self, pic, label,event_output_1,event_output_2):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic), F.to_tensor(label),F.to_tensor(event_output_1),F.to_tensor(event_output_2)




def train_dataloader_event(path, batch_size=64, num_workers=0, use_transform=True,pic_size=256):
    # image_dir = path
    image_dir = os.path.join(path,'train')

    transform = PairCompose_4(
        [
            PairRandomCrop_4(pic_size),
            PairRandomHorizontalFilp_4(),
            PairToTensor_4()
        ]
    )
    dataloader = DataLoader(
        DeblurDataset_With_Event_MAT(image_dir,transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

def valid_dataloader_event(path, batch_size=1, num_workers=0):
    image_dir = os.path.join(path,'valid')
    
    transform = PairCompose_4(
        [
            PairRandomCrop_4(360),
            PairRandomHorizontalFilp_4(),
            PairToTensor_4()
        ]
    )
    dataloader = DataLoader(
        DeblurDataset_With_Event_MAT(image_dir,transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader



def valid_dataloader_HQF(path, batch_size=1, num_workers=0):
    image_dir = os.path.join(path,'valid')
    

    dataloader = DataLoader(
        valid_DeblurDataset_HQF(image_dir),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader



class valid_DeblurDataset_HQF(udata.Dataset):
    def __init__(self, data_path,transform=None):
        super(valid_DeblurDataset_HQF, self).__init__()
        self.image_dir = data_path

        self.image_list = os.listdir(os.path.join(data_path, 'blur/'))
        self.image_list.sort()
        self.image_list = self.image_list[:-1]
        self.target_path = data_path
        self.input_path = data_path
        self.event_path = data_path

        target_path = os.path.join(self.target_path, 'H5/train_target.h5')
        input_path = os.path.join(self.input_path, 'H5/train_input.h5')

        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')

        self.keys = list(input_h5f.keys())
        random.shuffle(self.keys)
        target_h5f.close()
        input_h5f.close()


    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.image_dir, 'blur', self.image_list[idx]))  
        label = Image.open(os.path.join(self.image_dir, 'sharp', self.image_list[idx]))

        event_path = os.path.join(self.event_path,'H5/train_event_input.h5')

        event_h5f = h5py.File(event_path,'r')


        key = self.keys[idx]       
        event_output = np.array(event_h5f[key])     

        event_output = cv2.merge([event_output])    
        event_output = event_output.reshape(event_output.shape[1],event_output.shape[2],event_output.shape[0])
    
        event_output = cv2.resize(event_output,(image.size[1],image.size[0]))  

        event_output = event_output.reshape(event_output.shape[2],event_output.shape[1],event_output.shape[0])
        event_output_1 = event_output[0]
        event_output_2 = event_output[1]
        event_output_1 = Image.fromarray(event_output_1)
        event_output_2 = Image.fromarray(event_output_2)

        target = label
        input = image

        input = F.to_tensor(image)
        target = F.to_tensor(label)
        event_output_1 = F.to_tensor(event_output_1)
        event_output_2 = F.to_tensor(event_output_2)

        input = torch.cat((input,event_output_1),0)
        input = torch.cat((input,event_output_2),0)

        
        event_h5f.close()
        return input,target


class DeblurDataset_With_Event_MAT(udata.Dataset):
    def __init__(self, data_path,transform=None):
        super(DeblurDataset_With_Event_MAT, self).__init__()
        self.image_dir = data_path
        self.transform = transform
        self.image_list = os.listdir(os.path.join(data_path, 'blur/'))
        self.image_list.sort()
        self.image_list = self.image_list[:-1]
        self.target_path = data_path
        self.input_path = data_path
        self.event_path = data_path

        target_path = os.path.join(self.target_path, 'sharp/train_target.h5')
        input_path = os.path.join(self.input_path, 'blur/train_input.h5')

        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')

        self.keys = list(input_h5f.keys())
        random.shuffle(self.keys)
        target_h5f.close()
        input_h5f.close()

        
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.image_dir, 'blur', self.image_list[idx]))
        label = Image.open(os.path.join(self.image_dir, 'sharp', self.image_list[idx]))


        event_path = os.path.join(self.event_path,'H5/train_event_input.h5')
        event_h5f = h5py.File(event_path,'r')
        
        key = self.keys[idx]       
        event_output = np.array(event_h5f[key])  

        event_output = cv2.merge([event_output])   
        event_output = event_output.reshape(event_output.shape[1],event_output.shape[2],event_output.shape[0])

        event_output = cv2.resize(event_output,(image.size[1],image.size[0])) 
        event_output = event_output.reshape(event_output.shape[2],event_output.shape[0],event_output.shape[1])
        event_output_1 = event_output[0]
        event_output_2 = event_output[1]
        event_output_1 = Image.fromarray(event_output_1)
        event_output_2 = Image.fromarray(event_output_2)

        input, target,event_output_1,event_output_2 = self.transform(image, label,event_output_1,event_output_2)
        input = torch.cat((input,event_output_1),0)
        input = torch.cat((input,event_output_2),0)
        

        event_h5f.close()

        return input,target
