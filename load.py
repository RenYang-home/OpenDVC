import numpy as np
from scipy import misc

def load_data(data, frames, batch_size, Height, Width, Channel, folder, I_QP):

    for b in range(batch_size):

        path = folder[np.random.randint(len(folder))] + '/'

        bb = np.random.randint(0, 447 - 256)

        for f in range(frames):

            if f == 0:
                img = misc.imread(path + 'im1_bpg444_QP' + str(I_QP) + '.png')
                data[f, b, 0:Height, 0:Width, 0:Channel] = img[0:Height, bb: bb + Width, 0:Channel]
            else:
                img = misc.imread(path + 'im' + str(f + 1) + '.png')
                data[f, b, 0:Height, 0:Width, 0:Channel] = img[0:Height, bb: bb + Width, 0:Channel]

    return data


def load_data_ssim(data, frames, batch_size, Height, Width, Channel, folder, I_level):

    for b in range(batch_size):

        path = folder[np.random.randint(len(folder))] + '/'

        bb = np.random.randint(0, 447 - 256)

        for f in range(frames):

            if f == 0:
                img = misc.imread(path + 'im1_level' + str(I_level) + '_ssim.png')
                data[f, b, 0:Height, 0:Width, 0:Channel] = img[0:Height, bb: bb + Width, 0:Channel]
            else:
                img = misc.imread(path + 'im' + str(f + 1) + '.png')
                data[f, b, 0:Height, 0:Width, 0:Channel] = img[0:Height, bb: bb + Width, 0:Channel]

    return data

