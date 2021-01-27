# -*- coding: utf-8 -*-
"""
    main function to run AnimeGAN
    ~~~~~~~~~~~~~~~~~~~~~~

    :author: Xu Hangkun (许杭锟)
    :copyright: © 2020 Xu Hangkun <xuhangkun@163.com>
    :license: MIT, see LICENSE for more details.
"""

from AnimeGANv2.AnimeGAN import AnimeGAN
import argparse
import os
from AnimeGANv2.tools.utils import check_folder

def get_parser():
    parser = argparse.ArgumentParser(description="Parameters of AnimeGAN")
    parser.add_argument('--phase', type=str, default='train', help='train or test ?')
    parser.add_argument('--contain_init_phase',action='store_true', help='train phase')
    parser.add_argument('--input_size', type=list, default=[360,640], help='input image size [Height,width]')
    parser.add_argument('--style', type=str, default='Hayao', help='style name')

    parser.add_argument('--epoch', type=int, default=100, help='The number of epochs to run')
    parser.add_argument('--init_epoch', type=int, default=10, help='The number of epochs for weight initialization')
    parser.add_argument('--batch_size', type=int, default=8, help='The size of batch size') # if light : batch_size = 20
    parser.add_argument('--save_freq', type=int, default=1, help='The number of ckpt_save_freq')

    parser.add_argument('--init_lr', type=float, default=0.0002, help='The learning rate')
    parser.add_argument('--g_lr', type=float, default=0.0002, help='The learning rate')
    parser.add_argument('--d_lr', type=float, default=0.0002, help='The learning rate')

    parser.add_argument('--g_adv_weight', type=float, default=300.0, help='Weight about GAN')
    parser.add_argument('--d_adv_weight', type=float, default=300.0, help='Weight about GAN')
    parser.add_argument('--con_weight', type=float, default=1.5, help='Weight about VGG19')# 1.5 for Hayao, 2.0 for Paprika, 1.2 for Shinkai
    # ------ the follow weight used in AnimeGAN
    parser.add_argument('--sty_weight', type=float, default=2.5, help='Weight about style')# 2.5 for Hayao, 0.6 for Paprika, 2.0 for Shinkai
    parser.add_argument('--color_weight', type=float, default=10., help='Weight about color') # 15. for Hayao, 50. for Paprika, 10. for Shinkai
    parser.add_argument('--tv_weight', type=float, default=1., help='Weight about tv')# 1. for Hayao, 0.1 for Paprika, 1. for Shinkai

    parser.add_argument('--load_model',action='store_true',help='load model or create model from scratch')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    return parser

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    if args.phase == 'test':
        # --result_dir
        check_folder(args.result_dir)

    # --log_dir
    check_folder(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

"""main"""
def main():
    # parse arguments
    parser = get_parser()
    args = parser.parse_args()
    check_args(args)
    print(args)

    if args is None:
      exit()

    # open session
    gan = AnimeGAN(args)
    # show network architecture
    if args.phase == 'train':
        gan.train()
        gan.test()
    else:
        gan.test()

if __name__ == "__main__":
    main()
