import argparse
import os


def get_parser():
    parser = argparse.ArgumentParser(description='SSDU: Self-Supervision via Data Undersampling ')

    # %%parameters used for inference
    SSDU_RUN_TYPE = os.environ.get("SSDU_RUN_TYPE")
    if SSDU_RUN_TYPE == "INFERENCE":
        parser.add_argument('Inference_FID', type=str, 
                            help=' FID of the volunteer to go through network inference')

    # %% hyperparameters for the unrolled network
    parser.add_argument('--acc_rate', type=float, default=6.5,
                        help='acceleration rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--batchSize', type=int, default=1,
                        help='batch size')
    parser.add_argument('--nb_unroll_blocks', type=int, default=6,
                        help='number of unrolled blocks')
    parser.add_argument('--nb_res_blocks', type=int, default=15,
                        help="number of residual blocks in ResNet")
    parser.add_argument('--CG_Iter', type=int, default=20,
                        help='number of Conjugate Gradient iterations for DC')

    # %% hyperparameters for the dataset
    parser.add_argument('--data_opt', type=str, default='Coronal_PD',
                        help=' directories for the kspace, sensitivity maps and mask')
    parser.add_argument('--nrow_GLOB', type=int, default=414,
                        help='number of rows of the slices in the dataset')
    parser.add_argument('--ncol_GLOB', type=int, default=240,
                        help='number of columns of the slices in the dataset')
    parser.add_argument('--ncoil_GLOB', type=int, default=15,
                        help='number of coils of the slices in the dataset')

    # %% hyperparameters for the SSDU
    parser.add_argument('--mask_type', type=str, default='Gaussian',
                        help='mask selection for training and loss masks', choices=['Gaussian', 'Uniform'])
    parser.add_argument('--rho', type=float, default=0.4,
                        help='cardinality of the loss mask, \ rho = |\ Lambda| / |\ Omega|')
    parser.add_argument('--mu', type=float, default=0.01,
                        help='mu parameters for DC layer')
    parser.add_argument('--train_mode', type=str, default='SSDU',
                        help='modes to choose from \n SSDU: original mode \n FANSS: Dr.Fan s version of self-supervised learning',
                        choices=['SSDU', 'FANSS'])
    parser.add_argument('--weights', type=str, default='ssdu_original.h5',
                        help='weights name to save after trainning /or load during inference')

    return parser
