import argparse
import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
from scipy import misc
import CNN_img
import motion
import MC_network
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)

parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--ref", default='ref.png')
# parser.add_argument("--raw", default='raw.png')
parser.add_argument("--com", default='dec.png')
parser.add_argument("--bin", default='bitstream.bin')
parser.add_argument("--mode", default='PSNR', choices=['PSNR', 'MS-SSIM'])
# parser.add_argument("--metric", default='PSNR', choices=['PSNR', 'MS-SSIM'])
parser.add_argument("--l", type=int, default=1024, choices=[8, 16, 32, 64, 256, 512, 1024, 2048])
parser.add_argument("--N", type=int, default=128, choices=[128])
parser.add_argument("--M", type=int, default=128, choices=[128])

args = parser.parse_args()

batch_size = 1
Channel = 3

Y0_com_img = misc.imread(args.ref)
# Y1_raw_img = misc.imread(args.raw)

Y0_com_img = np.expand_dims(Y0_com_img, 0)
# Y1_raw_img = np.expand_dims(Y1_raw_img, 0)

Height = np.size(Y0_com_img, 1)
Width = np.size(Y0_com_img, 2)


Y0_com = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])
# Y1_raw = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])

string_mv_tensor = tf.placeholder(tf.string, [])
string_res_tensor = tf.placeholder(tf.string, [])

# Motion Decoding
entropy_bottleneck_mv = tfc.EntropyBottleneck(dtype=tf.float32, name='entropy_bottleneck')
flow_latent_hat = entropy_bottleneck_mv.decompress(
    tf.expand_dims(string_mv_tensor, 0), [Height//16, Width//16, args.M], channels=args.M)

# Residual Decoding
entropy_bottleneck_res = tfc.EntropyBottleneck(dtype=tf.float32, name='entropy_bottleneck_1_1')
res_latent_hat = entropy_bottleneck_res.decompress(
    tf.expand_dims(string_res_tensor, 0), [Height//16, Width//16, args.M], channels=args.M)


flow_hat = CNN_img.MV_synthesis(flow_latent_hat, args.N)

# Motion Compensation
Y1_warp = tf.contrib.image.dense_image_warp(Y0_com, flow_hat)

MC_input = tf.concat([flow_hat, Y0_com, Y1_warp], axis=-1)
Y1_MC = MC_network.MC(MC_input)

Res_hat = CNN_img.Res_synthesis(res_latent_hat, num_filters=args.N)

# Reconstructed frame
Y1_com = tf.clip_by_value(Res_hat + Y1_MC, 0, 1)

# if args.metric == 'PSNR':
#     train_mse = tf.reduce_mean(tf.squared_difference(Y1_com, Y1_raw))
#     quality = 10.0*tf.log(1.0/train_mse)/tf.log(10.0)
# elif args.metric == 'MS-SSIM':
#     quality = tf.math.reduce_mean(tf.image.ssim_multiscale(Y1_com, Y1_raw, max_val=1))

saver = tf.train.Saver(max_to_keep=None)
model_path = './OpenDVC_model/' + args.mode + '_' + str(args.l) + '_model/model.ckpt'
saver.restore(sess, save_path=model_path)

with open(args.bin, "rb") as ff:
    mv_len = np.frombuffer(ff.read(2), dtype=np.uint16)
    string_mv = ff.read(np.int(mv_len))
    string_res = ff.read()

compressed_frame = sess.run(Y1_com,
               feed_dict={Y0_com: Y0_com_img / 255.0,
                          # Y1_raw: Y1_raw_img / 255.0,
                          string_mv_tensor: string_mv,
                          string_res_tensor: string_res})


misc.imsave(args.com, np.uint8(np.round(compressed_frame[0] * 255.0)))

# print(args.metric + ' = ' + str(quality_com))
