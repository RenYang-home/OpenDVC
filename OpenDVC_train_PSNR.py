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
parser.add_argument("--l", type=int, default=1024, choices=[256, 512, 1024, 2048])
parser.add_argument("--N", type=int, default=128, choices=[128])
parser.add_argument("--M", type=int, default=128, choices=[128])
args = parser.parse_args()

if args.l == 256:
    I_QP = 37
elif args.l == 512:
    I_QP = 32
elif args.l == 1024:
    I_QP = 27
elif args.l == 2048:
    I_QP = 22

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
total_mse = tf.reduce_mean(tf.squared_difference(Y1_com, Y1_raw))
warp_mse = tf.reduce_mean(tf.squared_difference(Y1_warp, Y1_raw))
MC_mse = tf.reduce_mean(tf.squared_difference(Y1_raw, Y1_MC))

psnr = 10.0*tf.log(1.0/total_mse)/tf.log(10.0)

# The rate-distortion cost.
l = args.l

train_loss_total = l * total_mse + (train_bpp_MV + train_bpp_Res)
train_loss_MV = l * warp_mse + train_bpp_MV
train_loss_MC = l * MC_mse + train_bpp_MV

# Minimize loss and auxiliary loss, and execute update op.
step = tf.train.create_global_step()

train_MV = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_loss_MV, global_step=step)
train_MC = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_loss_MC, global_step=step)
train_total = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_loss_total, global_step=step)

aux_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate*10.0)
aux_step = aux_optimizer.minimize(entropy_bottleneck_mv.losses[0])

aux_optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate*10.0)
aux_step2 = aux_optimizer2.minimize(entropy_bottleneck_res.losses[0])

train_op_MV = tf.group(train_MV, aux_step, entropy_bottleneck_mv.updates[0])
train_op_MC = tf.group(train_MC, aux_step, entropy_bottleneck_mv.updates[0])
train_op_all = tf.group(train_total, aux_step, aux_step2,
                        entropy_bottleneck_mv.updates[0], entropy_bottleneck_res.updates[0])

tf.summary.scalar('psnr', psnr)
tf.summary.scalar('bits_total', train_bpp_MV + train_bpp_Res)
save_path = './OpenDVC_PSNR_' + str(l)
summary_writer = tf.summary.FileWriter(save_path, sess.graph)
saver = tf.train.Saver(max_to_keep=None)

sess.run(tf.global_variables_initializer())
var_motion = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='flow_motion')
saver_motion = tf.train.Saver(var_list=var_motion, max_to_keep=None)
saver_motion.restore(sess, save_path='motion_flow/model.ckpt-200000')

# Train
iter = 0

while(True):

    if iter <= 100000:
        frames = 2

        if iter <= 20000:
            train_op = train_op_MV
        elif iter <= 40000:
            train_op = train_op_MC
        else:
            train_op = train_op_all
    else:
        frames = 7
        train_op = train_op_all

    if iter <= 300000:
        lr = lr_init
    elif iter <= 600000:
        lr = lr_init / 10.0
    else:
        lr = lr_init / 100.0

    data = np.zeros([frames, batch_size, Height, Width, Channel])
    data = load.load_data(data, frames, batch_size, Height, Width, Channel, folder, I_QP)

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

        print('Training_OpenDVC Iteration:', iter)

        iter = iter + 1

        if iter % 500 == 0:

             merged_summary_op = tf.summary.merge_all()
             summary_str = sess.run(merged_summary_op, feed_dict={Y0_com: F0_com/255.0,
                                                                  Y1_raw: F1_raw/255.0})

             summary_writer.add_summary(summary_str, iter)

        if iter % 20000 == 0:

             checkpoint_path = os.path.join(save_path, 'model.ckpt')
             saver.save(sess, checkpoint_path, global_step=iter)

    if iter > 700000:
        break

    del data
    del F0_com
    del F1_raw
    del F1_decoded

    gc.collect()
