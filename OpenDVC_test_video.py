import argparse
import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
from scipy import misc
import CNN_img
import motion
import MC_network
import os
from ms_ssim_np import MultiScaleSSIM

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)

parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--path", default='BasketballPass')
parser.add_argument("--frame", type=int, default=100)
parser.add_argument("--GOP", type=int, default=10)
parser.add_argument("--mode", default='PSNR', choices=['PSNR', 'MS-SSIM'])
parser.add_argument("--metric", default='PSNR', choices=['PSNR', 'MS-SSIM'])
parser.add_argument("--python_path", default='path_to_python')
parser.add_argument("--CA_model_path", default='path_to_CA_EntropyModel_Test')
parser.add_argument("--l", type=int, default=1024, choices=[8, 16, 32, 64, 256, 512, 1024, 2048])
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

elif args.l == 8:
    I_level = 2
elif args.l == 16:
    I_level = 3
elif args.l == 32:
    I_level = 5
elif args.l == 64:
    I_level = 7

path = args.path + '/'
path_com = args.path + '_com_' + args.mode  + '_' + str(args.l) + '/'
path_bin = args.path + '_bin_' + args.mode  + '_' + str(args.l) + '/'

os.makedirs(path_com, exist_ok=True)
os.makedirs(path_bin, exist_ok=True)

batch_size = 1
Channel = 3

F1 = misc.imread(path + 'f001.png')
Height = np.size(F1, 0)
Width = np.size(F1, 1)

if (Height % 16 != 0) or (Width % 16 != 0):
    raise ValueError('Height and Width must be a mutiple of 16.')

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

quality_frame = np.zeros([args.frame])
bits_frame = np.zeros([args.frame])

for gop_number in range(np.int(np.ceil(args.frame/args.GOP))):

    f = gop_number * args.GOP

    if args.mode == 'PSNR':
        os.system('bpgenc -f 444 -m 9 ' + path + 'f' + str(f + 1).zfill(3) + '.png -o ' + path_bin + str(f + 1).zfill(3) + '.bin -q ' + str(I_QP))
        os.system('bpgdec ' + path_bin + str(f + 1).zfill(3) + '.bin -o ' + path_com + 'f' + str(f + 1).zfill(3) + '.png')
    elif args.mode == 'MS-SSIM':
        os.system(args.python_path + ' ' + args.CA_model_path + '/encode.py --model_type 1 --input_path ' + path + 'f' + str(f + 1).zfill(3) + '.png' +
                  ' --compressed_file_path ' + path_bin + str(f + 1).zfill(3) + '.bin' + ' --quality_level ' + str(I_level))
        os.system(args.python_path + ' ' + args.CA_model_path + '/decode.py --compressed_file_path ' + path_bin + str(f + 1).zfill(3) + '.bin'
                  + ' --recon_path ' + path_com + 'f' + str(f + 1).zfill(3) + '.png')

    bits = os.path.getsize(path_bin + str(f + 1).zfill(3) + '.bin')
    bits = bits * 8

    F0_com = misc.imread(path_com + 'f' + str(f + 1).zfill(3) + '.png')
    F0_raw = misc.imread(path + 'f' + str(f + 1).zfill(3) + '.png')

    F0_com = np.expand_dims(F0_com, axis=0)
    F0_raw = np.expand_dims(F0_raw, axis=0)

    if args.metric == 'PSNR':
        mse = np.mean(np.power(np.subtract(F0_com / 255.0, F0_raw / 255.0), 2.0))
        quality_frame[f] = 10 * np.log10(1.0 / mse)
    elif args.metric == 'MS-SSIM':
        quality_frame[f] = MultiScaleSSIM(F0_com, F0_raw, max_val=255)

    bits_frame[f] = bits
    print('Frame', f + 1, args.metric + ' =', quality_frame[f], 'bpp =', bits_frame[f] / Height / Width)

    p_frame_number = np.min([args.GOP - 1, args.frame - f - 1])

    for p_frame in range(p_frame_number):

        f = f + 1

        F1_raw = misc.imread(path + 'f' + str(f + 1).zfill(3) + '.png')
        F1_raw = np.expand_dims(F1_raw, axis=0)

        F0_com, string_MV, string_Res, quality_com \
            = sess.run([Y1_com, string_mv, string_res, quality],
                       feed_dict={Y0_com: F0_com / 255.0, Y1_raw: F1_raw / 255.0})

        F0_com = F0_com * 255.0

        with open(path_bin + str(f + 1).zfill(3) + '.bin', "wb") as ff:
            ff.write(np.array(len(string_MV), dtype=np.uint16).tobytes())
            ff.write(string_MV)
            ff.write(string_Res)

        misc.imsave(path_com + 'f' + str(f + 1).zfill(3) + '.png',
                    np.uint8(np.round(F0_com[0])))

        bits = (2 + len(string_MV) + len(string_Res)) * 8

        quality_frame[f] = quality_com
        bits_frame[f] = bits

        print('Frame', f + 1, args.metric + ' =', quality_frame[f], 'bpp =', bits_frame[f] / Height / Width)

quality_ave = np.average(quality_frame)
bits_ave = np.average(bits_frame / Height / Width)

print('Average ' + args.metric + ' =', quality_ave, 'Average bpp =', bits_ave)
