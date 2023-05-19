import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data_consistency as ssdu_dc
import tf_utils
import os


def display_output(ssdu_model, test_dataset, image_save_folder):
    for element in test_dataset.take(1):
        nw_input, sens_maps, trn_mask, _ = element[0]

        #display input image
        fig1 = plt.figure()
        nw_input_complex = tf_utils.tf_real2complex(nw_input).numpy()
        nw_input_complex = np.abs(np.squeeze(nw_input_complex))
        plt.imshow(nw_input_complex, cmap='gray')
        plt.colorbar()
        fig1.savefig(os.path.join(image_save_folder, 'nw_input.png'))

        #display network output image    
        output = ssdu_model(element[0], training = False)
        output = output[0]
        fig2 = plt.figure()
        output = tf_utils.tf_real2complex(output).numpy()
        output = np.abs(np.squeeze(output))
        plt.imshow(output, cmap='gray')
        plt.colorbar()
        fig2.savefig(os.path.join(image_save_folder,'raw_nw_output.png'))

        #display single dc_layer output image    
        mu = tf.constant(0., dtype=tf.float32)
        dc_out = ssdu_dc.dc_layer()([sens_maps, trn_mask, mu, nw_input])
        dc_out = tf_utils.tf_real2complex(dc_out).numpy()
        fig3 = plt.figure()
        plt.imshow(np.abs(np.squeeze(dc_out)), cmap='gray')
        plt.colorbar()
        fig3.savefig(os.path.join(image_save_folder,'dc_out.png'))

        #display EhE_OP result
        mu = tf.constant(0., dtype=tf.complex64)
        Encoder = ssdu_dc.data_consistency(sens_maps, trn_mask)
        EhEx = Encoder.EhE_Op(tf_utils.tf_real2complex(nw_input), mu)

        EhEx_complex = EhEx.numpy()
        EhEx_abs = np.abs(np.squeeze(EhEx_complex))

        fig4 = plt.figure()
        plt.imshow(np.abs(np.squeeze(EhEx_abs)), cmap='gray')
        plt.colorbar()
        fig4.savefig(os.path.join(image_save_folder,'EhE_out.png'))