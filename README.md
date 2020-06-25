# OpenDVC

An open source Tensorflow implementation of the paper:

Lu, Guo, et al. "DVC: An end-to-end deep video compression framework." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.

## Dependency

Tenforflow 1.12

Tensorflow-compression 1.0 (download at https://github.com/tensorflow/compression/releases/tag/v1.0)

Pre-trained models (download at https://drive.google.com/drive/folders/1gUkf9FNjiZw6Pcr5U_bl3jgbM1_ZpB2K?usp=sharing)

## How to use

We follow Lu et al., DVC to feed RBG images into the deep encoder. To compress a YUV video, please first convert to PNG images with the following command.

```
ffmpeg -pix_fmt yuv420p -s WidthxHeight -i  Name.yuv -vframes Frame path/f%03d.png
```
