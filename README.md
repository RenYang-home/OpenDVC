# OpenDVC

An open source Tensorflow implementation of the paper:

Lu, Guo, et al. "DVC: An end-to-end deep video compression framework." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2019.

## Dependency

Tenforflow 1.12

Tensorflow-compression 1.0 (download at https://github.com/tensorflow/compression/releases/tag/v1.0)

Pre-trained models (download at https://drive.google.com/drive/folders/1gUkf9FNjiZw6Pcr5U_bl3jgbM1_ZpB2K?usp=sharing)

BPG (download at https://bellard.org/bpg/) 

*In our PSNR model, we use BPG to compress I-frames instead of training learned image compression models

## How to use

We follow Lu et al., DVC to feed RGB images into the deep encoder. To compress a YUV video, please first convert to PNG images with the following command.

```
ffmpeg -pix_fmt yuv420p -s WidthxHeight -i  Name.yuv -vframes Frame path_to_PNG/f%03d.png
```
The pre-trained codes are trained by 4 lambda values (--l), i.e., 256, 512, 1024 and 2048, with increading bit-rate/PSNR. The test code for OpenDVC encoder is used as follows.
```
python OpenDVC_test_video.py --path path_to_PNG --l lambda
```

## Performance

We achieve comparable rate-distortion performance 
