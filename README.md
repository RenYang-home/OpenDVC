# OpenDVC -- An open source implementation of Deep Video Compression (DVC)

An open source Tensorflow implementation of the paper:

Lu, G., Ouyang, W., Xu, D., Zhang, X., Cai, C., & Gao, Z. (2019). DVC: An end-to-end deep video compression framework. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Contact:

Ren Yang @ ETH Zurich, Switzerland

Email: ren.yang@vision.ee.ethz.ch

## Dependency

Tenforflow 1.12

Tensorflow-compression 1.0 ([Download link](https://github.com/tensorflow/compression/releases/tag/v1.0))

Pre-trained models ([Download link](https://drive.google.com/drive/folders/1gUkf9FNjiZw6Pcr5U_bl3jgbM1_ZpB2K?usp=sharing))

BPG ([Download link](https://bellard.org/bpg/)) 

*In our PSNR model, we use BPG to compress I-frames instead of training learned image compression models

## How to use

We follow Lu *et al.*, DVC to feed RGB images into the deep encoder. To compress a YUV video, please first convert to PNG images with the following command.

```
ffmpeg -pix_fmt yuv420p -s WidthxHeight -i  Name.yuv -vframes Frame path_to_PNG/f%03d.png
```
The pre-trained codes are trained by 4 lambda values, i.e., 256, 512, 1024 and 2048, with increading bit-rate/PSNR. The test code for OpenDVC encoder is used as follows.
```
python OpenDVC_test_video.py --path path_to_PNG --l lambda
```
## To do

1. Release the pre-traied models optimized for MS-SSIM

2. Release the codes of decoder

## Performance

As shown in the figures below, our OpenDVC achieves comparable rate-distortion performance with the reported results in Lu *et al.*, DVC.
![ ](performance.pdf)

