from tools.adjust_brightness import adjust_brightness_from_src_to_dst, read_img
import os,cv2
import numpy as np
import torchvision.transforms as transforms
import torch


def load_test_data(image_path, size):
    img = cv2.imread(image_path).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocessing(img,size)
    img = np.expand_dims(img, axis=0)
    return img

def preprocessing(img, size):
    """
    scale the cv img to size
    convert the img to tensor
    convert the img tensor within (-1,1)
    """
    h, w = img.shape[:2]
    ratio = size*1.0 / h
    h = int(h*ratio)
    w = int(w*ratio)

    # the cv2 resize func : dsize format is (W ,H)
    img = cv2.resize(img, (w, h))
    img = np.asarray(img)
    # H x W x C -> C x H x W
    #img = np.transpose(img,[2,0,1])
    img = transforms.ToTensor()(img).unsqueeze(0)
    # preprocess, (-1, 1)
    img = -1 + 2 * img
    return img


def save_images(images, image_path, photo_path = None):
    fake = inverse_transform(images.squeeze())
    if photo_path:
        return imsave(adjust_brightness_from_src_to_dst(fake, read_img(photo_path)),  image_path)
    else:
        return imsave(fake, image_path)

def inverse_transform(images):
    images = (images + 1.) / 2 * 255
    # The calculation of floating-point numbers is inaccurate,
    # and the range of pixel values must be limited to the boundary,
    # otherwise, image distortion or artifacts will appear during display.
    images = np.clip(images, 0, 255)
    return images.astype(np.uint8)


def imsave(images, path):
    return cv2.imwrite(path, cv2.cvtColor(images, cv2.COLOR_BGR2RGB))

crop_image = lambda img, x0, y0, w, h: img[y0:y0+h, x0:x0+w]

def random_crop(img1, img2, crop_H, crop_W):

    assert  img1.shape ==  img2.shape
    h, w = img1.shape[:2]

    # The crop width cannot exceed the original image crop width
    if crop_W > w:
        crop_W = w

    # Crop height
    if crop_H > h:
        crop_H = h

    # Randomly generate the position of the upper left corner
    x0 = np.random.randint(0, w - crop_W + 1)
    y0 = np.random.randint(0, h - crop_H + 1)

    crop_1 = crop_image(img1, x0, y0, crop_W, crop_H)
    crop_2 = crop_image(img2, x0, y0, crop_W, crop_H)
    return crop_1,crop_2

def check_folder(log_dir):
    """
    check if the log_dir is existing
    if not existing, make the dir
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def str2bool(x):
    return x.lower() in ('true')


