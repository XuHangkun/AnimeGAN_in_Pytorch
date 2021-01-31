'''
   made by @finnkso (github)
   2020.04.09
   tensorflow-gpu==1.15.0  : tf.compat.v1
   if tensorflow-gpu==1.8.0, please replayce tf.compat.v1 to tf
'''
import argparse
import os
import tkinter as tk
from tkinter import filedialog
import cv2
from tqdm import tqdm
import numpy as np
import torch
from torch.autograd import Variable

from tools.utils import check_folder
from tools.adjust_brightness import adjust_brightness_from_src_to_dst


def parse_args():
    desc = "Make a style img"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--style',type=str,help="style")
    parser.add_argument('--video', type=str,help='video file or number for webcam')
    parser.add_argument('--output', type=str, default='dataset',help='relative output path')
    parser.add_argument('--start_index',type=int,default=1)
    parser.add_argument('--img_size',type=int,default=256)
    parser.add_argument('--start_time',type=float,default=120)
    parser.add_argument('--end_time',type=float,default=100000)
    parser.add_argument('--delta_time',type=float,default=1)

    return parser.parse_args()

def getfileloc(initialdir='/', method='open', title='Please select a file', filetypes=(("video files", ".mkv .avi .mp4"), ("all files","*.*"))):
    root = tk.Tk()
    if method == 'open':
        fileloc = filedialog.askopenfilename(parent=root, initialdir=initialdir, title=title, filetypes=filetypes)
    elif method == 'save':
        fileloc = filedialog.asksaveasfilename(parent=root, initialdir=initialdir, initialfile='out.avi', title=title, filetypes=filetypes)
    root.withdraw()
    return fileloc

def convert_image(img,size):
    """
    scale the cv img to size
    """
    h, w = img.shape[:2]
    img = img[int(h/10):int(9*h/10),int(w/10):int(9*w/10)]
    h, w = img.shape[:2]
    ratio = size*1.0 / h
    h = (int(h*ratio)//4)*4
    w = (int(w*ratio)//4)*4

    # the cv2 resize func : dsize format is (W ,H)
    img = cv2.resize(img, (w, h))
    return img

def save_anime_video_img(video,output,style,start_index=1,img_size=256,start_time=120.0,end_time=None,delta_time=1.0):
    '''
    Args:
        video : file path of video
        output: file path of output style image
        style : name of style
        ima_size: resize the image
        start_time(s): default = 120s
        end_time(s): default = None
        delta_time(s): 1
        start_index : start index of images
    Result:
        for eg: output = dataset, Style = Hayao
        save the style img in dataset/Hayao/style
        save the smoothed style img in dataset/Hayao/smooth/1.png
    '''

    # load video
    vid = cv2.VideoCapture(video)
    vid_name = os.path.basename(video)
    total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    START = start_time*fps
    EXTRACT_FREQUENCY = delta_time*fps

    # output path
    base_path = os.path.join(os.getcwd(),output,style)
    check_folder(base_path)
    style_path = os.path.join(base_path,"style")
    check_folder(style_path)
    smooth_path = os.path.join(base_path,"smooth")
    check_folder(smooth_path)

    # determine output width and height
    # shape of cv2 img H x W x C (BGR)
    ret, img = vid.read()
    if img is None:
        print('Error! Failed to determine frame size: frame empty.')
        return
    count=1
    save_index=start_index
    while ret:
        ret, frame = vid.read()
        count += 1
        if count < START :
            continue
        if frame is None:
            print('Warning: got empty frame.')
            continue
        if (count - START)%EXTRACT_FREQUENCY == 0:
            img = convert_image(frame, img_size)
            cv2.imwrite(os.path.join(style_path,"%d.png"%(save_index)),img)
            print(os.path.join(style_path,"%d.png"%(save_index)))
            save_index += 1
        if end_time and end_time*fps<count :
            break

if __name__ == '__main__':
    arg = parse_args()
    save_anime_video_img(arg.video,arg.output,style=arg.style,start_index=arg.start_index,img_size=arg.img_size,start_time=arg.start_time,end_time=arg.end_time)
