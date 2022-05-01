import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
import math
import re
import torch.nn as nn
import glob 

import scipy.io as scio
from PIL import Image

def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*epoch(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def normalize(data):
    return data / 255.

def is_image(img_name):
    if img_name.endswith(".jpg") or img_name.endswith(".bmp") or img_name.endswith(".png"):
        return True
    else:
        return  False

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0] 
    endw = img.shape[1] 
    endh = img.shape[2] 
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)

    for i in range(win):
        for j in range(win):
            patch = img[:, i:(endw - win + i + 1):stride, j:(endh - win + j + 1):stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

#####################################################################
#####################################################################
#####################################################################
def prepare_data_img(target_path,input_path):
    # train
    print('process training data')
    input_path = os.path.join(input_path,'blur')
    target_path = os.path.join(target_path,'sharp')

    target_list = os.listdir(target_path)
    target_list.sort()
    print('target_list=\n',target_list)
    input_list = os.listdir(input_path)
    input_list.sort()
    save_target_path = os.path.join('/gdata1/caocz/Deblur/HQF_Tiny/valid/H5', 'train_target.h5')
    save_input_path = os.path.join('/gdata1/caocz/Deblur/HQF_Tiny/valid/H5', 'train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    for i in range(len(target_list)):
        target_file = target_list[i]
        input_file = input_list[i]
        if os.path.exists(os.path.join(target_path,target_file)):


            img = Image.open(os.path.join(target_path,target_file))
            target_data = np.array(img)
            target_h5f.create_dataset(str(train_num), data=target_data)

            input_img = Image.open(os.path.join(input_path,input_file))
            input_img = np.array(input_img)
            input_h5f.create_dataset(str(train_num), data=input_img)

            train_num += 1
            print('i=',train_num)
            ##########################################################
    target_h5f.close()
    input_h5f.close()

    print('training set, # samples %d\n' % train_num)


#################################################################################
## Event
################################################################################33
def prepare_data_Event(input_path):
    # train
    print('process training data')
    input_path = os.path.join(input_path)

    input_list = os.listdir(input_path)
    input_list.sort()
    save_input_path = os.path.join('/gdata1/caocz/Deblur/HQF_Tiny/valid/H5', 'train_event_input.h5')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    for i in range(len(input_list)):
        input_file = input_list[i]
        if os.path.exists(os.path.join(input_path,input_file)):
            dataFile = os.path.join(input_path,input_file)
            event_sequence = scio.loadmat(dataFile)

            start_time=event_sequence['start_timestamp']
            end_time= np.array(event_sequence['section_event_timestamp'][-1][-1])
            end_time = np.expand_dims(end_time,axis=0)
            end_time = np.expand_dims(end_time,axis=0)
            event_time=event_sequence['section_event_timestamp']
            event_polar=event_sequence['section_event_polarity']
            event_y=event_sequence['section_event_y']
            event_x=event_sequence['section_event_x']


            image_interval=(end_time-start_time)
            frame_interval=(end_time-start_time)

            event_frame=np.zeros([2,260,346],int)
            for event_i in range(0,event_time.shape[1]):
                if event_time[0,event_i]<=image_interval+start_time:
                        cha = image_interval+start_time-event_time[0,event_i]
                        frame_index=cha//frame_interval
                        if frame_index==1:
                            frame_index=0
                        if event_polar[0,event_i]>0:
                            event_frame[int(frame_index*2)+1,event_y[0,event_i]-1,event_x[0,event_i]-1] = event_frame[int(frame_index*2)+1,event_y[0,event_i]-1,event_x[0,event_i]-1]+1
                        else:
                            event_frame[int(frame_index*2),event_y[0,event_i]-1,event_x[0,event_i]-1] = event_frame[int(frame_index*2),event_y[0,event_i]-1,event_x[0,event_i]-1]+1
                else:
                        cha = event_time[0,event_i]-image_interval-start_time
                        frame_index=cha//frame_interval
                        if frame_index==2:
                            frame_index=1
                        if event_polar[0,event_i]>0:
                            event_frame[int(frame_index*2),event_y[0,event_i]-1,event_x[0,event_i]-1] = event_frame[int(frame_index*2),event_y[0,event_i]-1,event_x[0,event_i]-1]+1
                        else:
                            event_frame[int(frame_index*2)+1,event_y[0,event_i]-1,event_x[0,event_i]-1] = event_frame[int(frame_index*2)+1,event_y[0,event_i]-1,event_x[0,event_i]-1]+1

            event_array=np.array(event_frame).astype(np.float32)

            input_h5f.create_dataset(str(train_num), data=event_array)


            train_num += 1

    input_h5f.close()
    print('training set, # samples %d\n' % train_num)


def read_data_Event(input_path):
    input_path = os.path.join(input_path, 'train_event_input.h5')
    target_h5f = h5py.File(input_path, 'r')

    keys = list(target_h5f.keys())
    output_polarity = np.array(target_h5f['0']).shape


prepare_data_img(target_path="",input_path='')
