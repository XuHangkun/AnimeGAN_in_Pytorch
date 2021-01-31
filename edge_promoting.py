import cv2, os
import numpy as np
from tqdm import tqdm
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Parameters of edge smooth")
    parser.add_argument('--style_dir', type=str, help='relative directory of style image')
    parser.add_argument('--smooth_dir', type=str, help='relative directory of smooth image')
    return parser

def edge_promoting(root, save):
    file_list = os.listdir(root)
    if not os.path.isdir(save):
        os.makedirs(save)
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss * gauss.transpose(1, 0)
    n = 1
    for f in tqdm(file_list):
        rgb_img = cv2.imread(os.path.join(root, f))
        gray_img = cv2.imread(os.path.join(root, f), 0)
        # rgb_img = cv2.resize(rgb_img, (256, 256))
        pad_img = np.pad(rgb_img, ((2,2), (2,2), (0,0)), mode='reflect')
        # gray_img = cv2.resize(gray_img, (256, 256))
        edges = cv2.Canny(gray_img, 100, 200)
        dilation = cv2.dilate(edges, kernel)

        gauss_img = np.copy(rgb_img)
        idx = np.where(dilation != 0)
        for i in range(np.sum(dilation != 0)):
            gauss_img[idx[0][i], idx[1][i], 0] = np.sum(np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0], gauss))
            gauss_img[idx[0][i], idx[1][i], 1] = np.sum(np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1], gauss))
            gauss_img[idx[0][i], idx[1][i], 2] = np.sum(np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2], gauss))

        #result = np.concatenate((rgb_img, gauss_img), 1)
        result = gauss_img

        cv2.imwrite(os.path.join(save, str(n) + '.png'), result)
        n += 1

if __name__ == "__main__":
    # parse aygument
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    base = os.getcwd()
    edge_promoting(os.path.join(base,args.style_dir),os.path.join(base,args.smooth_dir))
