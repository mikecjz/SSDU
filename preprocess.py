import numpy as np
import tensorflow as tf
import h5py
import os
import scipy.io
import random
import pandas as pd
import utils
from natsort import os_sorted


def get_enum_list(csv_path, tag, field_name = 'TrainValidationTest'):
    HOMEDIR = os.environ.get("HOME")
    SYSTYPE = os.environ.get("SYSTYPE")
    if SYSTYPE == "DISCOVERY":
        isDISCOVERY = True
    else:
        isDISCOVERY = False

    df = pd.read_csv(csv_path)
    filtered_df = df.loc[(df[field_name] == tag) & (df['rawPathOnServer'] == df['rawPathOnServer'])] # for NaN
    path_list = []

    for i in filtered_df['rawPathOnServer']:
        
        data_folder = os.path.join(i.split('/RAW')[0],'DL_data')

        if isDISCOVERY:
            data_folder = data_folder.replace('/home/jc_350', '/project')
        else:
            data_folder = data_folder.replace('/home/jc_350', HOMEDIR)

        for filename in os.listdir(data_folder):
            path_list.append(os.path.join(data_folder, filename))

    #Natural Sort
    path_list = os_sorted(path_list)

    #Enumeration
    enumerate_list = []
    for count,value in enumerate(path_list):
        enumerate_list.append((value,count))

    return enumerate_list
    
def loadmat_cart(matpath):
    f = scipy.io.loadmat(matpath) 
    
    
    MC_kspace = f['MC_kspace_slice']
    SE = f['SE']

    MC_kspace = MC_kspace.astype(np.complex64)
    SE = SE.astype(np.complex64)
    

    return MC_kspace, SE
            

# keyword args mask_train and mask_loss are named specifically not to be the same as
# the yielded trn_mask and loss_mask
def step_gen(data_list, original_mask, masker_object = None, shuffle = True, *, 
             mask_train = None, mask_loss = None):
    if shuffle:
        random.shuffle(data_list)
    for mat_path, count in data_list:
        # print('\nCase Nummmber =',count)
        print('\nMAT Path: ',mat_path)
        
        # MC_kspace should already be normalized between [0-1]
        MC_kspace, SE = loadmat_cart(mat_path)
        _, _, nCoil = np.shape(MC_kspace)

        # Use user supplied train mask
        if  mask_train != None:
            # print(mat_path, 'trainning mask specified')
            # if no loss mask is supplied, loss mask is the complement set of train mask
            if mask_loss == None:
                mask_loss = original_mask - mask_train
            trn_mask = mask_train
            loss_mask = mask_loss
        # Calculate train and loss mask on the fly
        else:
            # print(mat_path, 'making mask')
            trn_mask, loss_mask = masker_object.make_mask(MC_kspace, original_mask, mask_type='Gaussian')

        # print(mat_path, 'loss mask shape: ',loss_mask.shape)
        # print(mat_path, 'train mask shape: ', trn_mask.shape)
        # print(mat_path, 'original mask shape: ', original_mask.shape)
        sub_kspace = MC_kspace * np.tile(trn_mask[..., np.newaxis], (1, 1, nCoil))
        ref_kspace = MC_kspace * np.tile(loss_mask[..., np.newaxis], (1, 1, nCoil))

        nw_input = utils.sense1(sub_kspace, SE)

        # Prepare the data for the training
        SE = np.transpose(SE, (2, 3, 0, 1)) #(nCoil, nMaps, nRow, nCol)
        # print(mat_path, 'ref_kspace shape: ',ref_kspace.shape)
        # print(mat_path, 'MC_kspace shape:',MC_kspace.shape)
        
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
        

def step_gen_inference(data_list,original_mask,shuffle=False):
    for mat_path, count in data_list:
        print('\nCase Nummmber =',count)
        
        # MC_kspace should already be normalized between [0-1]
        MC_kspace, SE = loadmat_cart(mat_path)
        _, _, nCoil = np.shape(MC_kspace)

        sub_kspace = MC_kspace
        ref_kspace = MC_kspace

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
        trn_mask = tf.expand_dims(original_mask, axis = 0)
        loss_mask = tf.expand_dims(original_mask, axis = 0)

        multiple_inputs = (nw_input, SE, trn_mask, loss_mask)

        yield multiple_inputs, ref_kspace
     
