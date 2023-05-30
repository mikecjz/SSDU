from typing import Any
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Lambda, Add, Activation
from tensorflow.signal import fft2d, ifft2d, fftshift, ifftshift

import tf_utils
import parser_ops
import sys

parser = parser_ops.get_parser()
args = parser.parse_args()


class data_consistency():
    """
    Data consistency class can be used for:
        -performing E^h*E operation in the paper
        -transforming final network output to kspace
    """

    def __init__(self, sens_maps, mask):
        with tf.name_scope('EncoderParams'):
            self.shape_list = tf.shape(mask)
            self.sens_maps = sens_maps
            self.mask = mask
            self.shape_list = tf.shape(mask)
            self.scalar = tf.complex(tf.sqrt(tf.cast(self.shape_list[0] * self.shape_list[1], dtype=tf.float32)), 0.)

    def EhE_Op(self, img, mu):
        """
        Performs (E^h*E+ mu*I) x
        """
        with tf.name_scope('EhE'):
            #sense_maps: (1, nCoils, nMaps, nRow, nCol)
            #image:      (1, nMaps, nRow, nCol)
            coil_imgs = self.sens_maps * img #shape (1, nCoils, nMaps, nRow, nCol)
            kspace = fftshift(fft2d(ifftshift(coil_imgs,axes=(-1,-2))), axes=(-1,-2)) #/ self.scalar #shape (1, nCoils, nMaps, nRow, nCol)
            kspace = tf.reduce_sum(kspace, axis = -3) #shape (1, nCoils, nRow, nCol)

            masked_kspace = kspace * self.mask

            image_space_coil_imgs = ifftshift(ifft2d(fftshift(masked_kspace,axes=(-1,-2))), axes=(-1,-2)) #* self.scalar #shape (1, nCoils, nRow, nCol)
            image_space_coil_imgs = tf.expand_dims(image_space_coil_imgs, axis = 2) #shape (1, nCoils, 1, nRow, nCol)
            image_space_comb = tf.reduce_sum(image_space_coil_imgs * tf.math.conj(self.sens_maps), axis=-4) #shape (1, nMaps, nRow, nCol)

            ispace = image_space_comb + mu * img

        return ispace

    def SSDU_kspace(self, img):
        """
        Transforms unrolled network output to k-space
        and selects only loss mask locations(\Lambda) for computing loss
        """

        with tf.name_scope('SSDU_kspace'):
            #sense_maps: (1, nCoils, nMaps, nRow, nCol)
            #image:      (1, nMaps, nRow, nCol)
            coil_imgs = self.sens_maps * img #shape (1, nCoils, nMaps, nRow, nCol)
            kspace = fftshift(fft2d(ifftshift(coil_imgs, axes=(-1,-2))), axes=(-1,-2)) #/ self.scalar #shape (1, nCoils, nMaps, nRow, nCol)
            kspace = tf.reduce_sum(kspace, axis = -3) #shape (1, nCoils, nRow, nCol)
            masked_kspace = kspace * self.mask

        return masked_kspace

    def Supervised_kspace(self, img):
        """
        Transforms unrolled network output to k-space
        """

        with tf.name_scope('Supervised_kspace'):
            coil_imgs = self.sens_maps * img
            kspace = tf_utils.tf_fftshift(tf.signal.fft2d(tf_utils.tf_ifftshift(coil_imgs))) / self.scalar

        return kspace


def conj_grad(input_elems):
    """
    Parameters
    ----------
    input_data : contains tuple of  reg output rhs = E^h*y + mu*z , sens_maps and mask
    rhs = nrow x ncol x 2
    sens_maps : coil sensitivity maps ncoil x nrow x ncol
    mask : nrow x ncol
    mu : penalty parameter

    Encoder : Object instance for performing encoding matrix operations

    Returns
    -------
    data consistency output, nrow x ncol x 2

    """

    rhs, sens_maps, mask, mu_param = input_elems
    mu_param = tf.complex(mu_param, 0.)
    rhs = tf_utils.tf_real2complex(rhs)

    Encoder = data_consistency(sens_maps, mask)
    cond = lambda i, *_: tf.less(i, args.CG_Iter)

    def body(i, rsold, x, r, p, mu):
        with tf.name_scope('CGIters'):
            Ap = Encoder.EhE_Op(p, mu)
            alpha = tf.complex(rsold / tf.cast(tf.reduce_sum(tf.math.conj(p) * Ap), dtype=tf.float32), 0.)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = tf.cast(tf.reduce_sum(tf.math.conj(r) * r), dtype=tf.float32)
            beta = rsnew / rsold
            beta = tf.complex(beta, 0.)
            p = r + beta * p

        return i + 1, rsnew, x, r, p, mu

    x = tf.zeros_like(rhs)
    i, r, p = 0, rhs, rhs
    rsold = tf.cast(tf.reduce_sum(tf.math.conj(r) * r), dtype=tf.float32)
    loop_vars = i, rsold, x, r, p, mu_param
    cg_out = tf.while_loop(cond, body, loop_vars, name='CGloop', parallel_iterations=1)[2]

# %% CG Option 2

    # x = tf.zeros_like(rhs)
    # r, p = rhs, rhs
    # rsold = tf.cast(tf.reduce_sum(tf.math.conj(r) * r), dtype=tf.float32)

    # with tf.name_scope('CGIters'):
    #     for i in range(args.CG_Iter):
    #         Ap = Encoder.EhE_Op(p, mu_param)
    #         alpha = tf.complex(rsold / tf.cast(tf.reduce_sum(tf.math.conj(p) * Ap), dtype=tf.float32), 0.)
    #         x = x + alpha * p
    #         r = r - alpha * Ap
    #         rsnew = tf.cast(tf.reduce_sum(tf.math.conj(r) * r), dtype=tf.float32)
    #         beta = rsnew / rsold
    #         beta = tf.complex(beta, 0.)
    #         p = r + beta * p

    #         rsold = rsnew
    # cg_out = x
# %% CG return
    cg_out_real = tf_utils.tf_complex2real(cg_out)

    return cg_out_real

class dc_layer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        # self.sens_maps = sens_maps
        # self.mask = mask
        self.mu = self.add_weight(
            name="mu",
            shape=(1,),
            initializer='zeros',
            trainable=True
        )
    def call(self, input_elements):

        sens_maps, mask, nw_input_image, z = input_elements
        mu = self.mu

        rhs = nw_input_image + mu * z

        cg_out_real = conj_grad([rhs, sens_maps, mask, mu])
        
        return cg_out_real
    
def dc_block(rhs, sens_maps, mask, mu):
    """
    DC block employs conjugate gradient for data consistency,
    """
    dc_block_output  = Lambda(conj_grad)([rhs, sens_maps, mask, mu])

    return dc_block_output


def SSDU_kspace_transform(nw_output, sens_maps, mask):
    """
    This function transforms unrolled network output to k-space at only unseen locations in training (\Lambda locations)
    """

    nw_output = tf_utils.tf_real2complex(nw_output)

    def ssdu_map_fn(input_elems):
        nw_output_enc, sens_maps_enc, mask_enc = input_elems
        Encoder = data_consistency(sens_maps_enc, mask_enc)
        
        nw_output_kspace = Encoder.SSDU_kspace(nw_output_enc)

        return nw_output_kspace

    #masked_kspace = tf.map_fn(ssdu_map_fn, (nw_output, sens_maps, mask), dtype=tf.complex64, name='ssdumapFn')
    masked_kspace  = tf.keras.layers.Lambda(lambda x: ssdu_map_fn(x))((nw_output, sens_maps, mask))

    return tf_utils.tf_complex2real(masked_kspace)


def Supervised_kspace_transform(nw_output, sens_maps, mask):
    """
    This function transforms unrolled network output to k-space
    """

    nw_output = tf_utils.tf_real2complex(nw_output)

    def supervised_map_fn(input_elems):
        nw_output_enc, sens_maps_enc, mask_enc = input_elems
        Encoder = data_consistency(sens_maps_enc, mask_enc)
        nw_output_kspace = Encoder.Supervised_kspace(nw_output_enc)

        return nw_output_kspace

    kspace = tf.map_fn(supervised_map_fn, (nw_output, sens_maps, mask), dtype=tf.complex64, name='supervisedmapFn')

    return tf_utils.tf_complex2real(kspace)


