import tensorflow as tf
import scipy.io as sio
import numpy as np
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

parser = parser_ops.get_parser()
args = parser.parse_args()
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

save_dir ='saved_models'
directory = os.path.join(save_dir, 'SSDU_' + args.data_opt + '_' +str(args.epochs)+'Epochs_Rate'+ str(args.acc_rate) +\
                         '_' + str(args.nb_unroll_blocks) + 'Unrolls_' + args.mask_type+'Selection' )

if not os.path.exists(directory):
    os.makedirs(directory)

print('\n create a test model for the testing')
test_graph_generator = tf_utils.test_graph(directory)

#..............................................................................
start_time = time.time()
print('.................SSDU Training.....................')
tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

# .......................Load the Data..........................................
print('\n Loading ', args.data_opt, ' data, acc rate : ', args.acc_rate, ', mask type :', args.mask_type)
mask_dir = '/home/jc_350/zfan0804_712/Zhehao/Accelerated-VWI-Mask/mask_2x3_grappa.mat'

# %% kspace and sensitivity maps are assumed to be in .h5 format and mask is assumed to be in .mat
# Users can change these formats based on their dataset
original_mask = sio.loadmat(mask_dir)['mask']


# %%  zeropadded outer edges of k-space with no signal- check github readme file for explanation for further explanations
# for coronal PD dataset, first 17 and last 16 columns of k-space has no signal
# in the training mask we set corresponding columns as 1 to ensure data consistency

# %% set the batch size
# total_batch = int(np.floor(np.float32(nw_input.shape[0]) / (args.batchSize)))
# kspaceP = tf.placeholder(tf.float32, shape=(None, None, None, None, 2), name='refkspace')
# sens_mapsP = tf.placeholder(tf.complex64, shape=(None, None, None, None), name='sens_maps')
# trn_maskP = tf.placeholder(tf.complex64, shape=(None, None, None), name='trn_mask')
# loss_maskP = tf.placeholder(tf.complex64, shape=(None, None, None), name='loss_mask')
# nw_inputP = tf.placeholder(tf.float32, shape=(None, args.nrow_GLOB, args.ncol_GLOB, 2), name='nw_input')

print('Getting list of trainning files')
enumerate_trn_list = get_enum_list('dcm_tags_09_19_22.csv',tag='Train')
print('Getting list of trainning files done')
ssdu_masker = ssdu_masks.ssdu_masks()
total_batch = np.shape(enumerate_trn_list)[0]
dataset = tf.data.Dataset.from_generator(lambda: step_gen(enumerate_trn_list, original_mask, ssdu_masker,shuffle=True),
                                         output_types=((tf.float32, tf.float32, tf.complex64,
                                                           tf.complex64, tf.complex64))
                                        )
dataset = dataset.batch(args.batchSize)
dataset = dataset.prefetch(args.batchSize)
iterator = dataset.make_initializable_iterator()
ref_kspace_tensor, nw_input_tensor, sens_maps_tensor, trn_mask_tensor, loss_mask_tensor = iterator.get_next('getNext')
print(ref_kspace_tensor)
# %% make training model
nw_output_img, nw_output_kspace, *_ = UnrollNet.UnrolledNet(nw_input_tensor, sens_maps_tensor, trn_mask_tensor, loss_mask_tensor).model
scalar = tf.constant(0.5, dtype=tf.float32)
loss = tf.multiply(scalar, tf.norm(ref_kspace_tensor - nw_output_kspace) / tf.norm(ref_kspace_tensor)) + \
       tf.multiply(scalar, tf.norm(ref_kspace_tensor - nw_output_kspace, ord=1) / tf.norm(ref_kspace_tensor, ord=1))

all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss)

saver = tf.train.Saver(max_to_keep=100)
sess_trn_filename = os.path.join(directory, 'model')
totalLoss = []
avg_cost = 0
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    print('SSDU Parameters: Epochs: ', args.epochs, ', Batch Size:', args.batchSize,
          ', Number of trainable parameters: ', sess.run(all_trainable_vars))

    print('Training...')
    for ep in range(1, args.epochs + 1):
        sess.run(iterator.initializer)
        avg_cost = 0
        tic = time.time()
        try:
            for jj in range(total_batch):
                tmp, _, _ = sess.run([loss, update_ops, optimizer])
                avg_cost += tmp / total_batch
            toc = time.time() - tic
            totalLoss.append(avg_cost)
            print("Epoch:", ep, "elapsed_time =""{:f}".format(toc), "cost =", "{:.3f}".format(avg_cost))

        except tf.errors.OutOfRangeError:
            pass

        if (np.mod(ep, 10) == 0):
            saver.save(sess, sess_trn_filename, global_step=ep)
            sio.savemat(os.path.join(directory, 'TrainingLog.mat'), {'loss': totalLoss})

end_time = time.time()
sio.savemat(os.path.join(directory, 'TrainingLog.mat'), {'loss': totalLoss})
print('Training completed in  ', ((end_time - start_time) / 60), ' minutes')
