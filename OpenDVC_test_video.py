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
parser.add_argument("--path", default='BasketballPass')
parser.add_argument("--frame", type=int, default=100)
parser.add_argument("--GOP", type=int, default=10)
parser.add_argument("--l", type=int, default=1024)
parser.add_argument("--N", type=int, default=128)
parser.add_argument("--M", type=int, default=128)

args = parser.parse_args()

assert args.l == 256 or args.l == 512 or args.l == 1024 or 2048

if args.l == 256:
    I_QP = 37
elif args.l == 512:
    I_QP = 32
elif args.l == 1024:
    I_QP = 27
elif args.l == 2048:
    I_QP = 22

path_com = './' + args.path + '_com_' + str(args.l) + '/'
path_bin = './' + args.path + '_bin_' + str(args.l) + '/'

os.makedirs(path_com, exist_ok=True)
os.makedirs(path_bin, exist_ok=True)

batch_size = 1
Channel = 3

F1 = misc.imread(path + 'f001.png')
Height = np.size(F1, 0)
Width = np.size(F1, 1)

if (Height % 16 != 0) or (Width % 16 != 0):
    raise ValueError('Height and Width must be the mutiple of 16.')

Y0_com = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])
Y1_raw = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])

with tf.variable_scope("flow_motion"):

    flow_tensor, _, _, _, _, _ = motion.optical_flow(Y0_com, Y1_raw, batch_size, Height, Width)
    # Y1_warp_0 = tf.contrib.image.dense_image_warp(Y0_com, flow_tensor)

# Encode flow
mt = CNN_img.MV_analysis(flow_tensor, args.N, args.M)

entropy_bottleneck_mv = tfc.EntropyBottleneck()
string_mv = entropy_bottleneck_mv.compress(mt)
string_mv = tf.squeeze(string_mv, axis=0)

mt_hat, MV_likelihoods = entropy_bottleneck_mv(mt, training=False)

flow_hat = CNN_img.MV_synthesis(mt_hat, args.N)

# Motion Compensation
Y1_warp = tf.contrib.image.dense_image_warp(Y0_com, flow_hat)

MC_input = tf.concat([flow_hat, Y0_com, Y1_warp], axis=-1)
Y1_MC = MC_network.MC_new(MC_input)

# Encode residual
Res = Y1_raw - Y1_MC

y = CNN_img.Res_analysis(Res, num_filters=args.N, M=args.M)

entropy_bottleneck_res = tfc.EntropyBottleneck()
string_res = entropy_bottleneck_res.compress(y)
string_res = tf.squeeze(string_res, axis=0)

y_hat, Res_likelihoods = entropy_bottleneck_res(y, training=False)

Res_hat = CNN_img.Res_synthesis(y_hat, num_filters=args.N)

# Reconstructed frame
Y1_com = tf.clip_by_value(Res_hat + Y1_MC, 0, 1)

train_mse = tf.reduce_mean(tf.squared_difference(Y1_com, Y1_raw))
psnr = 10.0*tf.log(1.0/train_mse)/tf.log(10.0)

saver = tf.train.Saver(max_to_keep=None)
model_path = './PSNR_' + str(args.l) + '_model/model.ckpt'
saver.restore(sess, save_path=model_path)

psnr_frame = np.zeros([args.frame])
bits_frame = np.zeros([args.frame])

for gop_number in range(np.int(np.ceil(args.frame/args.GOP))):

    f = gop_number * args.GOP

    os.system('bpgenc -f 444 -m 9 ' + path + 'f' + str(f + 1).zfill(3) + '.png -o ' + path_bin + str(f + 1).zfill(3) + '.bin -q ' + str(I_QP))
    os.system('bpgdec ' + path_bin + str(f + 1).zfill(3) + '.bin -o ' + path_com + 'f' + str(f + 1).zfill(3) + '.png')
    bits = os.path.getsize(path_bin + str(f + 1).zfill(3) + '.bin')
    bits = bits * 8

    F0_com = misc.imread(path_com + 'f' + str(f + 1).zfill(3) + '.png')
    F0_raw = misc.imread(path + 'f' + str(f + 1).zfill(3) + '.png')
    mse = np.mean(np.power(np.subtract(F0_com / 255.0, F0_raw / 255.0), 2.0))

    psnr_frame[f] = 10 * np.log10(1.0 / mse)
    bits_frame[f] = bits
    print('Frame', f + 1, 'PSNR = ', psnr_frame[f], 'bpp = ', bits_frame[f] / Height / Width)

    F0_com = np.expand_dims(F0_com, axis=0)

    for p_frame in range(args.GOP - 1):

        f = f + 1

        F1_raw = misc.imread(path + 'f' + str(f + 1).zfill(3) + '.png')
        F1_raw = np.expand_dims(F1_raw, axis=0)

        F0_com, string_MV, string_Res, psnr_com \
            = sess.run([Y1_com, string_mv, string_res, psnr],
                       feed_dict={Y0_com: F0_com / 255.0, Y1_raw: F1_raw / 255.0})

        F0_com = F0_com * 255.0

        with open(path_bin + str(f + 1).zfill(3) + '.bin', "wb") as ff:
            ff.write(np.array(len(string_MV), dtype=np.uint16).tobytes())
            ff.write(string_MV)
            ff.write(string_Res)

        misc.imsave(path_com + 'f' + str(f + 1).zfill(3) + '.png',
                    np.uint8(np.round(F0_com[0])))

        bits = (2 + len(string_MV) + len(string_Res)) * 8

        psnr_frame[f] = psnr_com
        bits_frame[f] = bits

        print('Frame', f + 1, 'PSNR = ', psnr_frame[f], 'bpp = ', bits_frame[f] / Height / Width)

psnr_ave = np.average(psnr_frame)
bits_ave = np.average(bits_frame / Height / Width)

print('Average PSNR = ', psnr_ave, 'Average bpp = ', bits_ave)
