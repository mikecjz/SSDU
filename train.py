import numpy as np
import tensorflow as tf
import scipy.io as sio
import time
from datetime import datetime
import os
import h5py as h5
import utils
from preprocess import loadmat_cart, step_gen, get_enum_list
import tf_utils
import parser_ops
import masks.ssdu_masks as ssdu_masks
import UnrollNet
import matplotlib.pyplot as plt



parser = parser_ops.get_parser()
args = parser.parse_args()
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


#..............................................................................
start_time = time.time()

# .......................Load the Data..........................................
print('\n Loading ', args.data_opt, ' data, acc rate : ', args.acc_rate, ', mask type :', args.mask_type)
mask_dir = '/home/jc_350/zfan0804_712/Zhehao/Accelerated-VWI-Mask/mask_2x3_grappa.mat'
csv_dir = '/home/jc_350/DL/SSDU/dcm_tags_09_19_22.csv'

# %% kspace and sensitivity maps are assumed to be in .h5 format and mask is assumed to be in .mat
# Users can change these formats based on their dataset
original_mask = sio.loadmat(mask_dir)['mask']

print('Getting list of trainning files')
enumerate_trn_list = get_enum_list(csv_dir, tag='Train')
print('Getting list of trainning files done')

print('Getting list of validation files')
enumerate_val_list = get_enum_list(csv_dir, tag='Validation')
print('Getting list of validation files done')

#Setup Training and Loss Mask generator object
ssdu_masker = ssdu_masks.ssdu_masks()

# Setup trainning dataset 
train_dataset = tf.data.Dataset.from_generator(lambda: step_gen(enumerate_trn_list, original_mask, ssdu_masker,shuffle=True),
                                         output_types=((tf.float32, tf.complex64,
                                                           tf.complex64, tf.complex64), tf.float32 ))
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE) #pre_fetch for performance

# Setup validation dataset 
val_dataset = tf.data.Dataset.from_generator(lambda: step_gen(enumerate_val_list, original_mask, ssdu_masker,shuffle=True),
                                             output_types=((tf.float32, tf.complex64,
                                                           tf.complex64, tf.complex64), tf.float32 ))
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE) #pre_fetch for performance


test_dataset = tf.data.Dataset.from_generator(lambda: step_gen(enumerate_val_list[16:17], original_mask, ssdu_masker,shuffle=False),
                                         output_types=((tf.float32, tf.complex64,
                                                           tf.complex64, tf.complex64), tf.float32 ))
# %% make training model
ssdu_net= UnrollNet.UnrolledNet((args.nrow_GLOB, args.ncol_GLOB))
ssdu_model = ssdu_net.model

for element in test_dataset.take(1):

    fig1 = plt.figure()
    nw_input = utils.real2complex(element[0][0]).numpy()
    nw_input = np.abs(np.squeeze(nw_input))
    plt.imshow(nw_input, cmap='gray')
    plt.colorbar()
    fig1.savefig('nw_input.png')

    output = ssdu_model(element[0], training = False)
    output = output[0]
    fig2 = plt.figure()
    output = utils.real2complex(output).numpy()
    output = np.abs(np.squeeze(output))
    plt.imshow(output, cmap='gray')
    plt.colorbar()
    fig2.savefig('nw_dc_output.png')

    fig3 = plt.figure()
    plt.imshow(np.abs(output-nw_input), cmap='gray')
    plt.colorbar()
    fig3.savefig('difference.png')
ssdu_model_output_names = ssdu_model.output_names
print(ssdu_model_output_names)
# %% Setup trainning paramers
print('Setup trainning paramers')
custom_loss_mae_mse = utils.MAE_MSE_LOSS(lam=0.5)
mse_loss = tf.keras.losses.MeanSquaredError()
dummy_loss = utils.dummy_loss
opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

print('Compiling Model')
ssdu_model.compile(optimizer=opt, loss=[dummy_loss, custom_loss_mae_mse, dummy_loss])
# %% Setup logging and callbacks
print('Setup logging and callbacks')

logdir = os.path.join("logs/Unrolled/", datetime.now().strftime("%Y%m%d-%H%M%S"))
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()

tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1,
                                                        update_freq='batch')

checkpoint_filepath = "/home/jc_350/DL/Checkpoints/ssdu_best.h5"
# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_filepath,
#     save_weights_only=True,
#     monitor='custom_loss_mae_mse',
#     verbose=1,
#     mode='min',
#     save_best_only=True)

# %% Train Model
print('Training Model')
history = ssdu_model.fit(train_dataset, epochs=100, steps_per_epoch=96,
                                 validation_data=val_dataset,validation_steps=80,
                                 callbacks=[tensorboard_callback])

ssdu_model.save_weights('ssdu_original.h5')