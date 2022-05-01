# STRA

<img src= "https://github.com/Chengzhi-Cao/STRA/tree/main/img/network.jpg" width="80%">

This repository provides the official PyTorch implementation of the following paper:

> Event-driven Video Deblurring via Spatio-Temporal Relation-Aware Network
>
> Chengzhi Cao, Xueyang Fu*, Yurui Zhu, Gege Shi, Zheng-jun Zha
>
> In IJCAI 2022.
>
> Paper Link:
>
> Video deblurring with event information has attracted considerable attention. To help deblur each frame, existing methods usually compress a specific event sequence into a feature tensor with the same size as the corresponding video. However, this strategy neither considers the pixel-level spatial brightness changes nor the temporal correlation between events at each time step, resulting in insufficient use of spatio-temporal information. To address this issue, we propose a new Spatio-Temporal Relation-Attention network (STRA), for the specific event-based video deblurring. Concretely, to utilize spatial consistency between the frame and event, we model the brightness changes as an extra prior to aware blurring contexts in each frame; to record temporal relationship among different events, we develop a temporal memory block to restore long-range dependencies of event sequences continuously. In this way, the complementary information contained in the events and frames, as well as the correlation of neighboring events, can be fully utilized to recover spatial texture from events constantly. Experiments show that our STRA significantly outperforms several competing methods, e.g., on the HQF dataset, our network achieves up to 1.3 dB in terms of PSNR over the most advanced method.

---

## Contents

The contents of this repository are as follows:

1. [Dependencies](#Dependencies)
2. [Dataset](#Dataset)
3. [Train](#Train)
4. [Test](#Test)

---

## Dependencies

- Python
- Pytorch (1.4)
- scikit-image
- opencv-python

---

## Dataset

- Download deblur dataset from the [GoPro dataset](https://seungjunnah.github.io/Datasets/gopro.html) .

- Unzip files ```dataset``` folder.

- Preprocess dataset by running the command below:

  ``` python data/preprocessing.py```

After preparing data set, the data folder should be like the format below:

```
GOPRO
├─ train
│ ├─ blur    % 2103 image pairs
│ │ ├─ xxxx.png
│ │ ├─ ......
│ │
│ ├─ sharp
│ │ ├─ xxxx.png
│ │ ├─ ......
│
├─ test    % 1111 image pairs
│ ├─ ...... (same as train)

```

---

## Train

To train STRA , run the command below:

``` python main.py --model_name "STRA" --mode "train" --data_dir "dataset/GOPRO" ```

Model weights will be saved in ``` results/model_name/weights``` folder.

---

## Test

To test STRA , run the command below:

``` python main.py --model_name "STRA" --mode "test" --data_dir "dataset/GOPRO" --test_model "xxx.pkl" ```

Output images will be saved in ``` results/model_name/result_image``` folder.

---

## Contact
Should you have any question, please contact chengzhicao@mail.ustc.edu.cn.

## Notes and references
The  code is based on the MIMO-UNet project: https://github.com/chosj95/mimo-unet
