from __future__ import print_function, absolute_import
import torch.utils.data as data
import os
import numpy as np
import cv2
import sys

this_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(this_dir, "../../"))
from lib.utils.utils import random_img_aug
import random


class _W_Pad(data.Dataset):
    """"固定H和W训练，但是W不是通过resize成同一尺寸（会导致文字变形），而是通过padding的方式"""
    def __init__(self, config, is_train=True, image_aug=False):

        self.root = config.DATASET.ROOT
        self.is_train = is_train
        self.image_aug = image_aug
        self.inp_h = config.MODEL.IMAGE_SIZE.H
        self.inp_w = config.MODEL.IMAGE_SIZE.W

        self.dataset_name = config.DATASET.DATASET

        self.mean = np.array(config.DATASET.MEAN, dtype=np.float32)
        self.std = np.array(config.DATASET.STD, dtype=np.float32)

        char_file = config.DATASET.CHAR_FILE
        # with open(char_file, 'rb') as file:
        #     char_dict = {num: char.strip().decode('gbk', 'ignore') for num, char in enumerate(file.readlines())}
        with open(char_file, 'r', encoding="utf-8") as file:
            char_dict = {num: char.strip() for num, char in enumerate(file.readlines())}
            # char_dict = {}
            # for num, char in enumerate(file.readlines()):
            #     char = char.strip()
            #     if type(char) == bytes:
            #         import pdb; pdb.set_trace()
            #         print(char)
            #     char_dict[num] = char.strip()

        txt_file = config.DATASET.JSON_FILE['train'] if is_train else config.DATASET.JSON_FILE['val']

        # convert name:indices to name:string
        self.labels = []
        with open(txt_file, 'r', encoding='utf-8') as file:
            contents = file.readlines()
            for c in contents:
                imgname = c.strip().split(' ')[0]
                indices = c.strip().split(' ')[1:]

                string = ''.join([char_dict[int(idx)] for idx in indices])
                self.labels.append({imgname: string})

        print("load {} images!".format(self.__len__()))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        img_name = list(self.labels[idx].keys())[0]
        img = cv2.imread(os.path.join(self.root, img_name))

        if self.is_train and self.image_aug:  # 训练过程中随机拓增数据
            random_time = random.randint(0, 2)
            try:
                img = random_img_aug(img, random_time)  
            except:
                # import pdb; pdb.set_trace()
                print(img_name)
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        except:
            # import pdb; pdb.set_trace()
            print(img_name)

        img = gray_img_pad_resize(img, self.inp_h, self.inp_w)

        # img_h, img_w = img.shape
        # img = cv2.resize(img, (0,0), fx=self.inp_h / img_h, fy=self.inp_h / img_h, interpolation=cv2.INTER_CUBIC)  # 只固定高，保持文字不形变

        img = np.reshape(img, (self.inp_h, -1, 1))

        img = img.astype(np.float32)
        img = (img/255. - self.mean) / self.std
        img = img.transpose([2, 0, 1])

        return img, idx

def gray_img_pad_resize(img, target_h, target_w):
    """图像放在左上角，其余补零"""
    output_img = np.zeros((target_h, target_w))
    img_h, img_w = img.shape
    if img_w / img_h > target_w / target_h:  # 图像非常长，需要压缩h<32
        w_new = target_w
        h_new = int((img_h/img_w) * w_new)
    else:  # 图像正常长，让h尽可能沾满
        h_new = target_h
        w_new = int((img_w/img_h) * h_new)
    img_new = cv2.resize(img, (w_new, h_new))

    output_img[:h_new, :w_new] = img_new

    return output_img


