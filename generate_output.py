import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data_consistency as ssdu_dc
import tf_utils
import os
import scipy.io


def display_output(ssdu_model, test_dataset, image_save_folder):

    def myprint(s):
        with open(os.path.join(image_save_folder, 'modelsummary.txt'),'a') as f:
            print(s, file=f)

    if os.path.exists(os.path.join(image_save_folder, 'modelsummary.txt')):
        os.remove(os.path.join(image_save_folder, 'modelsummary.txt'))

    ssdu_model.summary(print_fn=myprint)
    for element in test_dataset.take(1):
        nw_input, sens_maps, trn_mask, loss_mask = element[0]

        #display input image
        fig1 = plt.figure()
        nw_input_complex = tf_utils.tf_real2complex(nw_input).numpy()[0,0,...]
        nw_input_complex = np.abs(np.squeeze(nw_input_complex))
        plt.imshow(nw_input_complex, cmap='gray')
        plt.colorbar()
        fig1.savefig(os.path.join(image_save_folder, 'nw_input.png'))

        #display network output image    
        output = ssdu_model.predict(element[0])
        output = output[0]
        fig2 = plt.figure()
        output = tf_utils.tf_real2complex(output).numpy()[0, 0,...]
        output = np.abs(np.squeeze(output))
        plt.imshow(output, cmap='gray')
        plt.colorbar()
        fig2.savefig(os.path.join(image_save_folder,'raw_nw_output.png'))

        #display single dc_layer output image    
        dc_out = ssdu_dc.dc_layer()([sens_maps, trn_mask, nw_input, tf.zeros_like(nw_input)])
        dc_out = tf_utils.tf_real2complex(dc_out).numpy()[0, 0,...]
        fig3 = plt.figure()
        plt.imshow(np.abs(np.squeeze(dc_out)), cmap='gray')
        plt.colorbar()
        fig3.savefig(os.path.join(image_save_folder,'dc_out.png'))

        #display EhE_OP result
        mu = tf.constant(0., dtype=tf.complex64)
        Encoder = ssdu_dc.data_consistency(sens_maps, trn_mask)
        EhEx = Encoder.EhE_Op(tf_utils.tf_real2complex(nw_input), mu)

        EhEx_complex = EhEx.numpy()[0, 0,...]
        EhEx_abs = np.abs(np.squeeze(EhEx_complex))

        fig4 = plt.figure()
        plt.imshow(np.abs(np.squeeze(EhEx_abs)), cmap='gray')
        plt.colorbar()
        fig4.savefig(os.path.join(image_save_folder,'EhE_out.png'))

        #save train and loss mask for tesing
        mask_dict = {"trn_mask": np.squeeze(trn_mask.numpy()), "loss_mask":  np.squeeze(loss_mask.numpy())}
        scipy.io.savemat(os.path.join(image_save_folder,'trn_loss_masks.mat'), mask_dict)

def SSDU_inference(ssdu_model, inference_dataset, patient_folder, num_slices=350, inference_folder='DL_inference/SSDU'):
    DL_inference_folder = os.path.join(patient_folder, inference_folder)
    os.makedirs(DL_inference_folder, exist_ok=True)

    iSlice = 1
    for element in inference_dataset.take(num_slices):
        print('Performing Inference for Slice ', str(iSlice), '. Patient folder:  ', patient_folder)
        #get network output image    
        output = ssdu_model.predict(element[0])
        output = output[0]
        output = tf_utils.tf_real2complex(output).numpy()[0, ...] #shape (1, nMaps, nRow, nCol)
        output = np.squeeze(output) #shape (nMaps, nRow, nCol)
        output = np.transpose(output, (1, 2, 0)) #shape (nRow, nCol, nMaps)

        save_dict = {"SC_12_SSDU": output}
        scipy.io.savemat(os.path.join(DL_inference_folder,'Slice_' + str(iSlice) + '.mat'), save_dict)
        iSlice = iSlice+1
            