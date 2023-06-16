import numpy as np
import tensorflow as tf
import scipy.io as sio
import time
from datetime import datetime
import os
os.environ["SSDU_RUN_TYPE"] = "INFERENCE"
import h5py as h5
import utils
from preprocess import loadmat_cart, step_gen, get_enum_list, step_gen_inference
import data_consistency as ssdu_dc
import tf_utils
import parser_ops
import masks.ssdu_masks as ssdu_masks
import pandas as pd
import UnrollNet
import matplotlib.pyplot as plt
from  generate_output import display_output, SSDU_inference


parser = parser_ops.get_parser()
args = parser.parse_args()
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


HOMEDIR = os.environ.get("HOME")
SYSTYPE = os.environ.get("SYSTYPE")
working_dir = os.path.join(HOMEDIR, 'DL/SSDU')


if SYSTYPE == "DISCOVERY":
    zfan0804_712_dir = '/project/zfan0804_712'
    isDISCOVERY = True
else:
    zfan0804_712_dir = os.path.join(HOMEDIR, 'zfan0804_712')
    isDISCOVERY = False

#..............................................................................
start_time = time.time()

# .......................Load the Data..........................................
print('\n Loading ', args.Inference_FID, ' for inference \n')
print('working dir: ', HOMEDIR)
print('SYSTYPE: ', str(SYSTYPE))
print('zfan0804_712_dir: ',zfan0804_712_dir)
mask_dir = os.path.join(zfan0804_712_dir, 'Zhehao/Accelerated-VWI-Mask/mask_2x3_grappa.mat')
csv_dir = os.path.join(working_dir, 'dcm_tags_09_19_22.csv')

# %% kspace and sensitivity maps are assumed to be in .h5 format and mask is assumed to be in .mat
# Users can change these formats based on their dataset
original_mask = sio.loadmat(mask_dir)['mask']

print('Getting list of inference files')
enumerate_inference_list = get_enum_list(csv_dir, tag=args.Inference_FID, field_name='FID')
print('Getting list of inference files done')


#Setup Training and Loss Mask generator object
ssdu_masker = ssdu_masks.ssdu_masks()

# Setup inference dataset 
inference_dataset = tf.data.Dataset.from_generator(lambda: step_gen_inference(enumerate_inference_list, original_mask, shuffle=False),
                                         output_types=((tf.float32, tf.complex64,
                                                           tf.complex64, tf.complex64), tf.float32 ))

# %% load trained model
ssdu_net= UnrollNet.UnrolledNet((args.nrow_GLOB, args.ncol_GLOB))
ssdu_model = ssdu_net.model
weights_h5_name = args.weights
ssdu_model.load_weights(os.path.join(working_dir, weights_h5_name))

# %% Inference
df = pd.read_csv(csv_dir)
filtered_df = df.loc[(df['FID'] == args.Inference_FID) & (df['rawPathOnServer'] == df['rawPathOnServer'])] # for NaN

raw_data_path = filtered_df["rawPathOnServer"]

patient_folder = raw_data_path.values[0].split('/RAW')[0]

if isDISCOVERY:
    patient_folder = patient_folder.replace('/home/jc_350', '/project')
else:
    patient_folder = patient_folder.replace('/home/jc_350', HOMEDIR)
SSDU_inference(ssdu_model, inference_dataset, patient_folder)
