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
from net.generator_cartoon import Generator
from torch.autograd import Variable

from tools.utils import preprocessing,check_folder
from tools.adjust_brightness import adjust_brightness_from_src_to_dst


def parse_args():
    desc = "Convert video to cartoon movie"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--load_size', default = 720)
    parser.add_argument('--model_path', type=str, default='./checkpoint/CartoonGAN',
                        help='directory of model')
    parser.add_argument('--style', default = 'Hayao')
    parser.add_argument('--video', type=str, default='./video/input/test.avi',
                        help='video file or number for webcam')
    parser.add_argument('--output', type=str, default='./video/output/',
                        help='output path')
    parser.add_argument('--output_format', type=str, default='MP4V',
                        help='codec used in VideoWriter when saving video to file')
    parser.add_argument('--gpu', type=int, default = 1)
    """
    output_format: xxx.mp4('MP4V'), xxx.mkv('FMP4'), xxx.flv('FLV1'), xxx.avi('XIVD')
    ps. ffmpeg -i xxx.mkv -c:v libx264 -strict -2 xxxx.mp4, this command can convert mkv to mp4, which has small size.
    """

    return parser.parse_args()

def getfileloc(initialdir='/', method='open', title='Please select a file', filetypes=(("video files", ".mkv .avi .mp4"), ("all files","*.*"))):
    root = tk.Tk()
    if method == 'open':
        fileloc = filedialog.askopenfilename(parent=root, initialdir=initialdir, title=title, filetypes=filetypes)
    elif method == 'save':
        fileloc = filedialog.asksaveasfilename(parent=root, initialdir=initialdir, initialfile='out.avi', title=title, filetypes=filetypes)
    root.withdraw()
    return fileloc



def convert_image(img, img_size):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocessing(img, img_size)
    return img

def inverse_image(img):
    img = (img.squeeze().data.cpu().float()+1.) / 2 * 255
    img = np.transpose(img,[1,2,0])
    return img

def cvt2anime_video(video, output, model_dir,style,gpu=1,output_format='MP4V', img_size=720):
    '''
    output_format: 4-letter code that specify codec to use for specific video type. e.g. for mp4 support use "H264", "MP4V", or "X264"
    '''
    # load pretrained model
    model = torch.load(os.path.join(model_dir,style+'/%s_generator.pth'%(style)))
    model.eval()
    if gpu > -1:
        print('GPU mode')
        model.cuda()
    else:
        print('CPU mode')
        model.float()

    # load video
    vid = cv2.VideoCapture(video)
    vid_name = os.path.basename(video)
    total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    # codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    codec = cv2.VideoWriter_fourcc(*output_format)

    # determine output width and height
    # shape of cv2 img H x W x C (BGR)
    ret, img = vid.read()
    if img is None:
        print('Error! Failed to determine frame size: frame empty.')
        return
    img = preprocessing(img, img_size)
    height, width = img.shape[2:]
    # out = cv2.VideoWriter(os.path.join(output, vid_name.replace('mp4','mkv')), codec, fps, (width, height))
    out = cv2.VideoWriter(os.path.join(output, vid_name), codec, fps, (width, height))
    pbar = tqdm(total=total)
    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while ret:
        ret, frame = vid.read()
        if frame is None:
            print('Warning: got empty frame.')
            continue
        img = convert_image(frame, img_size)
        if gpu > -1:
        	img = Variable(img, volatile=True).cuda()
        else:
            img = Variable(img, volatile=True).float()
        fake_img = model(img)
        fake_img = inverse_image(fake_img)
        fake_img = adjust_brightness_from_src_to_dst(fake_img, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        out.write(cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB))
        pbar.update(1)
    pbar.close()
    vid.release()
    # cv2.destroyAllWindows()
    return os.path.join(output, vid_name)


if __name__ == '__main__':
    arg = parse_args()
    if not arg.video:
        arg.video = getfileloc(initialdir='input/')
    else:
        arg.video = os.path.join(os.path.dirname(os.path.dirname(__file__)), arg.video)
    if not arg.output:
        arg.output = getfileloc(initialdir='output/', method='save')
    else:
        arg.output = os.path.join(os.path.dirname(os.path.dirname(__file__)), arg.output)
    check_folder(arg.output)
    info = cvt2anime_video(arg.video, arg.output, model_dir = arg.model_path,style=arg.style, output_format=arg.output_format)
    print(f'output video: {info}')
