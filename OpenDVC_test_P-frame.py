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
parser.add_argument("--raw", default='raw.png')
parser.add_argument("--com", default='com.png')
parser.add_argument("--bin", default='bitstream.bin')
parser.add_argument("--mode", default='PSNR', choices=['PSNR', 'MS-SSIM'])
parser.add_argument("--metric", default='PSNR', choices=['PSNR', 'MS-SSIM'])
parser.add_argument("--l", type=int, default=1024, choices=[8, 16, 32, 64, 256, 512, 1024, 2048])
parser.add_argument("--N", type=int, default=128, choices=[128])
parser.add_argument("--M", type=int, default=128, choices=[128])

args = parser.parse_args()

batch_size = 1
Channel = 3

Y0_com_img = misc.imread(args.ref)
Y1_raw_img = misc.imread(args.raw)

Y0_com_img = np.expand_dims(Y0_com_img, 0)
Y1_raw_img = np.expand_dims(Y1_raw_img, 0)

Height = np.size(Y1_raw_img, 1)
Width = np.size(Y1_raw_img, 2)

Y0_com = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])
Y1_raw = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])

with tf.variable_scope("flow_motion"):

    flow_tensor, _, _, _, _, _ = motion.optical_flow(Y0_com, Y1_raw, batch_size, Height, Width)
    # Y1_warp_0 = tf.contrib.image.dense_image_warp(Y0_com, flow_tensor)

# Encode flow
flow_latent = CNN_img.MV_analysis(flow_tensor, args.N, args.M)

entropy_bottleneck_mv = tfc.EntropyBottleneck()
string_mv = entropy_bottleneck_mv.compress(flow_latent)
string_mv = tf.squeeze(string_mv, axis=0)

flow_latent_hat, MV_likelihoods = entropy_bottleneck_mv(flow_latent, training=False)

flow_hat = CNN_img.MV_synthesis(flow_latent_hat, args.N)

# Motion Compensation
Y1_warp = tf.contrib.image.dense_image_warp(Y0_com, flow_hat)

MC_input = tf.concat([flow_hat, Y0_com, Y1_warp], axis=-1)
Y1_MC = MC_network.MC(MC_input)

# Encode residual
Res = Y1_raw - Y1_MC

res_latent = CNN_img.Res_analysis(Res, num_filters=args.N, M=args.M)

entropy_bottleneck_res = tfc.EntropyBottleneck()
string_res = entropy_bottleneck_res.compress(res_latent)
string_res = tf.squeeze(string_res, axis=0)

res_latent_hat, Res_likelihoods = entropy_bottleneck_res(res_latent, training=False)

Res_hat = CNN_img.Res_synthesis(res_latent_hat, num_filters=args.N)

# Reconstructed frame
Y1_com = tf.clip_by_value(Res_hat + Y1_MC, 0, 1)

if args.metric == 'PSNR':
    train_mse = tf.reduce_mean(tf.squared_difference(Y1_com, Y1_raw))
    quality = 10.0*tf.log(1.0/train_mse)/tf.log(10.0)
elif args.metric == 'MS-SSIM':
    quality = tf.math.reduce_mean(tf.image.ssim_multiscale(Y1_com, Y1_raw, max_val=1))

saver = tf.train.Saver(max_to_keep=None)
model_path = './OpenDVC_model/' + args.mode + '_' + str(args.l) + '_model/model.ckpt'
saver.restore(sess, save_path=model_path)

compressed_frame, string_MV, string_Res, quality_com \
    = sess.run([Y1_com, string_mv, string_res, quality],
               feed_dict={Y0_com: Y0_com_img / 255.0, Y1_raw: Y1_raw_img / 255.0})

with open(args.bin, "wb") as ff:
    ff.write(np.array(len(string_MV), dtype=np.uint16).tobytes())
    ff.write(string_MV)
    ff.write(string_Res)

misc.imsave(args.com, np.uint8(np.round(compressed_frame[0] * 255.0)))
bpp = (2 + len(string_MV) + len(string_Res)) * 8 / Height / Width

print(args.metric + ' = ' + str(quality_com), 'bpp = ' + str(bpp))
