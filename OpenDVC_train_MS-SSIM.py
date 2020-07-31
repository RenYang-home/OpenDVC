import argparse
import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
import os
import CNN_img
import motion
import MC_network
import load
import gc

config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)

parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--l", type=int, default=32, choices=[8, 16, 32, 64])
parser.add_argument("--N", type=int, default=128, choices=[128])
parser.add_argument("--M", type=int, default=128, choices=[128])
args = parser.parse_args()

if args.l == 8:
    I_level = 2
elif args.l == 16:
    I_level = 3
elif args.l == 32:
    I_level = 5
elif args.l == 64:
    I_level = 7

batch_size = 4
Height = 256
Width = 256
Channel = 3
lr_init = 1e-4

folder = np.load('folder.npy')

Y0_com = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])
Y1_raw = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])
learning_rate = tf.placeholder(tf.float32, [])

with tf.variable_scope("flow_motion"):

    flow_tensor, _, _, _, _, _ = motion.optical_flow(Y0_com, Y1_raw, batch_size, Height, Width)
    # Y1_warp_0 = tf.contrib.image.dense_image_warp(Y0_com, flow_tensor)

# Encode flow
flow_latent = CNN_img.MV_analysis(flow_tensor, args.N, args.M)

entropy_bottleneck_mv = tfc.EntropyBottleneck()
string_mv = entropy_bottleneck_mv.compress(flow_latent)
# string_mv = tf.squeeze(string_mv, axis=0)

flow_latent_hat, MV_likelihoods = entropy_bottleneck_mv(flow_latent, training=True)

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
# string_res = tf.squeeze(string_res, axis=0)

res_latent_hat, Res_likelihoods = entropy_bottleneck_res(res_latent, training=True)

Res_hat = CNN_img.Res_synthesis(res_latent_hat, num_filters=args.N)

# Reconstructed frame
Y1_com = Res_hat + Y1_MC

# Total number of bits divided by number of pixels.
train_bpp_MV = tf.reduce_sum(tf.log(MV_likelihoods)) / (-np.log(2) * Height * Width * batch_size)
train_bpp_Res = tf.reduce_sum(tf.log(Res_likelihoods)) / (-np.log(2) * Height * Width * batch_size)

# Mean squared error across pixels.
frame_msssim = tf.math.reduce_mean(tf.image.ssim_multiscale(Y1_com, Y1_raw, max_val=1))

# The rate-distortion cost.
l = args.l

train_loss_total = l * (1 - frame_msssim) + (train_bpp_MV + train_bpp_Res)

# Minimize loss and auxiliary loss, and execute update op.
step = tf.train.create_global_step()

train_total = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_loss_total, global_step=step)

aux_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate*10.0)
aux_step = aux_optimizer.minimize(entropy_bottleneck_mv.losses[0])

aux_optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate*10.0)
aux_step2 = aux_optimizer2.minimize(entropy_bottleneck_res.losses[0])

train_op = tf.group(train_total, aux_step, aux_step2,
                        entropy_bottleneck_mv.updates[0], entropy_bottleneck_res.updates[0])

tf.summary.scalar('ms-ssim', frame_msssim)
tf.summary.scalar('bits_total', train_bpp_MV + train_bpp_Res)
save_path = './OpenDVC_MS-SSIM_' + str(l)

summary_writer = tf.summary.FileWriter(save_path, sess.graph)
saver = tf.train.Saver(max_to_keep=None)

saver_psnr = tf.train.Saver(max_to_keep=None)
latest = tf.train.latest_checkpoint(checkpoint_dir='./OpenDVC_PSNR_' + str(l * 32))
saver_psnr.restore(sess, save_path=latest)

# Train
iter = 0

while(True):

    frames = 7

    if iter <= 200000:
        lr = lr_init
    else:
        lr = lr_init / 10.0

    data = np.zeros([frames, batch_size, Height, Width, Channel])
    data = load.load_data_ssim(data, frames, batch_size, Height, Width, Channel, folder, I_level)

    for ff in range(frames-1):

        if ff == 0:

            F0_com = data[0]
            F1_raw = data[1]

            _, F1_decoded = sess.run([train_op, Y1_com],
                                     feed_dict={Y0_com: F0_com / 255.0,
                                                Y1_raw: F1_raw / 255.0,
                                                learning_rate: lr})

        else:

            F0_com = F1_decoded * 255.0
            F1_raw = data[ff+1]

            _, F1_decoded = sess.run([train_op, Y1_com],
                                     feed_dict={Y0_com: F0_com / 255.0,
                                                Y1_raw: F1_raw / 255.0,
                                                learning_rate: lr})

        print('Fine-tuning_OpenDVC_MS-SSIM Iteration:', iter)

        iter = iter + 1

        if iter % 500 == 0:

             merged_summary_op = tf.summary.merge_all()
             summary_str = sess.run(merged_summary_op, feed_dict={Y0_com: F0_com/255.0,
                                                                  Y1_raw: F1_raw/255.0})

             summary_writer.add_summary(summary_str, iter)

        if iter % 20000 == 0:

             checkpoint_path = os.path.join(save_path, 'model.ckpt')
             saver.save(sess, checkpoint_path, global_step=iter)

    if iter > 300000:
        break

    del data
    del F0_com
    del F1_raw
    del F1_decoded

    gc.collect()
