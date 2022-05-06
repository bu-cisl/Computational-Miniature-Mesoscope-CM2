from __future__ import print_function
import os
from cm2_models import Demixer_ResNet, Reconstructor_ResNet
import numpy as np
import tifffile
import tensorflow as tf

tf.keras.mixed_precision.experimental.set_policy('float16')

from tensorflow.keras.layers import Lambda, Input, Concatenate
from tensorflow.keras.models import Model
from skimage import io

def refocusing_and_crop(vars):
    lf_data = vars[0]
    padding = vars[1]
    rows = vars[2]
    cols = vars[3]
    lf_data = tf.reshape(lf_data, [-1, rows + 2 * padding, cols + 2 * padding, 3, 3])
    # lf_data = tf.transpose(lf_data, perm=[0, 1, 2, 4, 3])
    rfv_list = []
    for shift in range(-17, 19, 1):  ## TBD
        tmp = lf_data[:, :, :, 1, 1]
        tmp = tmp + tf.roll(lf_data[:, :, :, 0, 0], shift=[shift, shift], axis=[1, 2])
        tmp = tmp + tf.roll(lf_data[:, :, :, 0, 1], shift=shift, axis=1)
        tmp = tmp + tf.roll(lf_data[:, :, :, 0, 2], shift=[shift, -shift], axis=[1, 2])
        tmp = tmp + tf.roll(lf_data[:, :, :, 1, 0], shift=shift, axis=2)
        tmp = tmp + tf.roll(lf_data[:, :, :, 1, 2], shift=-shift, axis=2)
        tmp = tmp + tf.roll(lf_data[:, :, :, 2, 0], shift=[-shift, shift], axis=[1, 2])
        tmp = tmp + tf.roll(lf_data[:, :, :, 2, 1], shift=-shift, axis=1)
        tmp = tmp + tf.roll(lf_data[:, :, :, 2, 2], shift=[-shift, -shift], axis=[1, 2])
        tmp = tmp / 9.0
        tmp = tf.expand_dims(tmp, axis=-1)
        rfv_list.append(tmp)
    rfv = tf.concat(rfv_list, axis=-1)
    return rfv[:, padding:-padding, padding:-padding, :]


tot_len = 1920
kernel_size = 3
padding = 32
rows = tot_len - 2 * padding
cols = rows

demixer = Demixer_ResNet(rows + 2 * padding, cols + 2 * padding, 3, 64, 16)  # tbd
reconstructor = Reconstructor_ResNet(rows, cols, 9, 'add', 36, 80, kernel_size, 64, 16, 16, None)
input_views = Input(shape=[rows + 2 * padding, cols + 2 * padding, 9], name='input_views')
demixed_views = demixer(input_views)
rfv = Lambda(refocusing_and_crop)([demixed_views, padding, rows, cols])
crop_dmx_views = Lambda(lambda x: x[:, padding:-padding, padding:-padding, :])(demixed_views)
views_with_rfv = Concatenate(axis=-1)([crop_dmx_views, rfv])
pred_vol = reconstructor(views_with_rfv)
joint_model = Model(inputs=[input_views], outputs=[demixed_views, rfv, pred_vol])
joint_model.load_weights('cm2net.hdf5')

# choose reference for experimental data (histogram matching))
# ref = io.imread('.tif')
# ref = ref.astype('float32') / 65535.0

# used on synthetic data
loc = np.array(
    [[392, 916], [393, 1562], [394, 2206], [1032, 918], [1032, 1561], [1032, 2204], [1681, 917], [1680, 1561],
     [1679, 2203]])

# set your result path
results_dir = 'result/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

meas = io.imread('measurement.tif')
meas = meas.astype('float64') / 65535.0
# meas = match_histograms(meas, ref, multichannel=False)
tmp_pad = 900
meas = np.pad(meas, ((tmp_pad, tmp_pad), (tmp_pad, tmp_pad)), 'constant', constant_values=0)
x = np.zeros((1, tot_len, tot_len, 9), 'float16')
for k in range(9):
    x[0, :, :, k] = (meas[loc[k, 0] - (tot_len // 2) + tmp_pad:loc[k, 0] + (tot_len // 2) + tmp_pad,
                     loc[k, 1] - (tot_len // 2) + tmp_pad:loc[k, 1] + (tot_len // 2) + tmp_pad]).astype('float16')
[pred_dmx, pred_rfv, pred_rec] = joint_model.predict(x, batch_size=1)

pred_dmx = (np.transpose(pred_dmx[0, :].squeeze(), [2, 0, 1]) * 65535.0).astype('uint16')
pred_dmx = np.pad(pred_dmx, ((0, 0), (padding, padding), (padding, padding)), 'constant',
                  constant_values=0)
tifffile.imwrite((results_dir + '/dmx.tif'), pred_dmx)

pred_rfv = (np.transpose(pred_rfv[0, :].squeeze(), [2, 0, 1]) * 65535.0).astype('uint16')
pred_rfv = np.pad(pred_rfv, ((0, 0), (padding, padding), (padding , padding)), 'constant',
                  constant_values=0)
tifffile.imwrite((results_dir + '/rfv.tif'), pred_rfv)

pred_rec = pred_rec / pred_rec.max()
pred_rec = (np.transpose(pred_rec[0, :].squeeze(), [2, 0, 1]) * 65535.0).astype('uint16')
pred_rec = np.pad(pred_rec, ((0, 0), (padding, padding), (padding, padding)), 'constant',
                  constant_values=0)
tifffile.imwrite(results_dir + '/rec.tif', pred_rec)

