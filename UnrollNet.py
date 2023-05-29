import tensorflow as tf
import data_consistency as ssdu_dc
import tf_utils
import models.networks as networks
import parser_ops

from tensorflow.keras.models import Model
from tensorflow.keras import Input

parser = parser_ops.get_parser()
args = parser.parse_args("")


class UnrolledNet():
    """

    Parameters
    ----------
    input_x: batch_size x nrow x ncol x 2
    sens_maps: batch_size x ncoil x nrow x ncol

    trn_mask: batch_size x nrow x ncol, used in data consistency units
    loss_mask: batch_size x nrow x ncol, used to define loss in k-space

    args.nb_unroll_blocks: number of unrolled blocks
    args.nb_res_blocks: number of residual blocks in ResNet

    Returns
    ----------

    x: nw output image
    nw_kspace_output: k-space corresponding nw output at loss mask locations

    x0 : dc output without any regularization.
    all_intermediate_results: all intermediate outputs of regularizer and dc units
    mu: learned penalty parameter


    """

    def __init__(self, input_size):
        self.input_x = Input((2,) + input_size + (2,), dtype=tf.float32, name='nw_input')
        self.sens_maps = Input((None, 2) + input_size, dtype=tf.complex64, name='sens_maps')
        self.trn_mask = Input(input_size, dtype=tf.complex64, name='train_mask')
        self.loss_mask = Input(input_size, dtype=tf.complex64, name='loss_mask')
        self.model = self.Unrolled_SSDU()

    def Unrolled_SSDU(self):
        x = self.input_x
        all_intermediate_results = [[0 for _ in range(2)] for _ in range(args.nb_unroll_blocks)]

        mu = tf.Variable(0., dtype=tf.float32)
        x0 = ssdu_dc.dc_layer()([self.sens_maps, self.trn_mask, mu, x])
    
        with tf.name_scope('SSDUModel'):
            for i in range(args.nb_unroll_blocks):
                x = networks.ResNet(x, args.nb_res_blocks)
                denoiser_output = x

                rhs = self.input_x + mu * x

                x = ssdu_dc.dc_layer()([self.sens_maps, self.trn_mask, mu, rhs])
                dc_output = x
                # ...................................................................................................
                all_intermediate_results[i][0] = tf_utils.tf_real2complex(tf.squeeze(denoiser_output))
                all_intermediate_results[i][1] = tf_utils.tf_real2complex(tf.squeeze(dc_output))

        nw_kspace_output = ssdu_dc.SSDU_kspace_transform(x, self.sens_maps, self.loss_mask)
        print('Model output shape' + str(nw_kspace_output.shape))
        # model = Model(inputs=[self.input_x, self.sens_maps, self.trn_mask, self.loss_mask], 
        #                outputs={'image_output': x, 'kspace_output': nw_kspace_output, 'first_sense_image': x0, 'intermediate_results': all_intermediate_results}) 
        model = Model(inputs=[self.input_x, self.sens_maps, self.trn_mask, self.loss_mask], 
                      outputs=[dc_output, nw_kspace_output, x0])    

        return model
