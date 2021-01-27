# -*- coding: utf-8 -*-
"""
    1. Calculate the (mB,mG,mR) channel mean value  of an image dataset
    2. Calculate the mean value of all channel of an image dateset
    3. Return (mA-mR,mA-mG,mA-rB)
    ~~~~~~~~~~~~~~~~~~~~~~

    :author: Xu Hangkun (许杭锟)
    :copyright: © 2020 Xu Hangkun <xuhangkun@163.com>
    :license: MIT, see LICENSE for more details.
"""

import cv2, argparse, os
from glob import glob
from tqdm import tqdm

def parse_args():
    desc = "Get the mean values of  R,G,B on the whole dataset"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset', type=str, default='Paprika', help='dataset_name')

    return parser.parse_args()

def read_img(image_path):
    img = cv2.imread(image_path)
    assert len(img.shape) == 3
    B = img[..., 0].mean()
    G = img[..., 1].mean()
    R = img[..., 2].mean()
    return B,G,R

def get_mean(dataset_name):
    file_list = glob(os.path.dirname(os.path.dirname(__file__))+'/dataset/{}/{}/*.*'.format(dataset_name, 'style'))
    image_num = len(file_list)
    print('image_num:', image_num)

    B_total = 0
    G_total = 0
    R_total = 0
    for f in tqdm(file_list) :
        bgr = read_img(f)
        B_total += bgr[0]
        G_total += bgr[1]
        R_total += bgr[2]

    B_mean, G_mean, R_mean = B_total / image_num, G_total / image_num, R_total / image_num
    mean = (B_mean + G_mean + R_mean)/3

    return mean-R_mean, mean-G_mean, mean-B_mean

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    return get_mean(args.dataset)
if __name__ == '__main__':
    result = main()
    print('style_data_mean_diff (B, G, R):  ', result)