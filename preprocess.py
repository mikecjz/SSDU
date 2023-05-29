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
        print('\nCase Nummmber =',count)
        
        # MC_kspace should already be normalized between [0-1]
        MC_kspace, SE = loadmat_cart(mat_path)
        _, _, nCoil = np.shape(MC_kspace)

        
         
        trn_mask, loss_mask = ssdu_masker.make_mask(MC_kspace, original_mask, mask_type='Gaussian')

        sub_kspace = MC_kspace * np.tile(trn_mask[..., np.newaxis], (1, 1, nCoil))
        ref_kspace = MC_kspace * np.tile(loss_mask[..., np.newaxis], (1, 1, nCoil))

        nw_input = utils.sense1(sub_kspace, SE)

        # Prepare the data for the training
        SE = np.transpose(SE, (2, 3, 0, 1)) #(nCoil, nMaps, nRow, nCol)
        ref_kspace = utils.complex2real(np.transpose(ref_kspace, (2, 0, 1))) #(nCoil, nRow, nCol)
        nw_input = np.transpose(nw_input, (2, 0, 1)) #(nMaps, nRow, nCol)

        nw_input = utils.complex2real(nw_input)

        #Expland first dimension to 1 for batch dimension
        ref_kspace = tf.expand_dims(ref_kspace, axis = 0)
        nw_input = tf.expand_dims(nw_input, axis = 0)
        SE = tf.expand_dims(SE, axis = 0)
        trn_mask = tf.expand_dims(trn_mask, axis = 0)
        loss_mask = tf.expand_dims(loss_mask, axis = 0)
        
        # print('ref_kspace shape' + str(ref_kspace.shape))
        # print('nw_input shape' + str(nw_input.shape))
        # print('SE shape' + str(SE.shape))
        # print('trn_mask shape' + str(trn_mask.shape))
        # print('loss_mask shape' + str(loss_mask.shape)) 


        multiple_inputs = (nw_input, SE, trn_mask, loss_mask)

        yield multiple_inputs, ref_kspace
        


     
