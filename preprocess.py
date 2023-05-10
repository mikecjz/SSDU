import numpy as np
import tensorflow as tf
import h5py
import os
import scipy.io
import random
import pandas as pd
import utils

def get_enum_list(csv_path, tag):
    df = pd.read_csv(csv_path)
    filtered_df = df.loc[(df['TrainValidationTest'] == tag) & (df['rawPathOnServer'] == df['rawPathOnServer'])] # for NaN
    path_list = []

    for i in filtered_df['rawPathOnServer']:
        
        data_folder = os.path.join(i.split('/RAW')[0],'DL_data')
        for filename in os.listdir(data_folder):
            path_list.append(os.path.join(data_folder, filename))

    #Enumeration
    enumerate_list = []
    for count,value in enumerate(path_list):
        enumerate_list.append((value,count))

    return enumerate_list
    
def loadmat_cart(matpath):
    f = scipy.io.loadmat(matpath) 
    
    
    MC_kspace = f['MC_kspace_slice'][()].view(np.complex64)
    SE = f['SE'][()].view(np.complex64)
    

    return MC_kspace, SE
            
        
def step_gen(data_list,original_mask,ssdu_masker,shuffle=True):
    if shuffle:
        random.shuffle(data_list)
    for mat_path, count in data_list:
        print('\nCase Number =',count)
        
        # MC_kspace should already be normalized between [0-1]
        MC_kspace, SE = loadmat_cart(mat_path)
        _, _, nCoil = np.shape(MC_kspace)

        trn_mask, loss_mask = ssdu_masker.make_mask(MC_kspace, original_mask, mask_type='Gaussian')

        sub_kspace = MC_kspace * np.tile(trn_mask[..., np.newaxis], (1, 1, nCoil))
        ref_kspace = MC_kspace * np.tile(loss_mask[..., np.newaxis], (1, 1, nCoil))

        nw_input = utils.sense1(sub_kspace, SE)

        # Prepare the data for the training
        SE = np.transpose(SE, (0, 3, 1, 2))
        ref_kspace = utils.complex2real(np.transpose(ref_kspace, (0, 3, 1, 2)))
        nw_input = utils.complex2real(nw_input)

        # Convert to tf arrays
        ref_kspace = tf.convert_to_tensor(ref_kspace)
        nw_input = tf.convert_to_tensor(nw_input)
        SE = tf.convert_to_tensor(SE)
        trn_mask = tf.convert_to_tensor(trn_mask)
        loss_mask = tf.convert_to_tensor(loss_mask)

        yield ref_kspace, nw_input, SE, trn_mask, loss_mask
        


     
