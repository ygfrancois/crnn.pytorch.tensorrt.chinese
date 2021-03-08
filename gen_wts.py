import torch
from torch.autograd import Variable
import yaml
from easydict import EasyDict as edict
import sys, os
import lib.models.crnn as CRNN_model
import lib.config.alphabets as alphabets
import lib.utils.utils as utils

import struct


def init_args():
    import argparse
    parser = argparse.ArgumentParser(description="gen wts from pytorch weights")

    ## CRNN
    parser.add_argument("--crnn_cfg", help="experiment configuration filename", type=str, default='lib/config/cn_config.yaml')
    parser.add_argument("--ocr_recognition_model_path", default='weights/crnn_cn.pth', type=str)
    # parser.add_argument("--alphabet_path", default='lib/config/alphabet_6863.list', type=str)
    parser.add_argument("--wts_save_path", help="wts filepath transformed from torch pth", type=str, default='./crnn_trt/crnn.wts')

    args = parser.parse_args()
    return args




if __name__ == "__main__":
    args = init_args()
    # 文字识别模型
    with open(args.crnn_cfg, "r") as f:
        config_crnn = yaml.load(f)
        config_crnn = edict(config_crnn)

        if config_crnn.DATASET.CHAR_FILE:
            # alphabets_char_file = [char.split('\n')[0] for char in open(config_crnn.DATASET.CHAR_FILE).readlines()[1:]]
            alphabets_char_file = [char.strip() for char in open(config_crnn.DATASET.CHAR_FILE).readlines()[1:]]
            alphabets_char_file = ''.join(alphabets_char_file)
            config_crnn.DATASET.ALPHABETS = alphabets_char_file
        else:
            config_crnn.DATASET.ALPHABETS = alphabets.alphabet

        # alphabets_char_file = [char.split('\n')[0] for char in open(args.alphabet_path).readlines()[1:]]
        # alphabets_char_file = "".join(alphabets_char_file)
        # config_crnn.DATASET.ALPHABETS = alphabets_char_file

        config_crnn.MODEL.NUM_CLASSES = len(config_crnn.DATASET.ALPHABETS)        

    ocr_rec_model = CRNN_model.get_crnn(config_crnn).to(0)
    print("loading pretrained rec model from {0}".format(args.ocr_recognition_model_path))
    checkpoint = torch.load(args.ocr_recognition_model_path)
    if "state_dict" in checkpoint.keys():
        ocr_rec_model.load_state_dict(checkpoint["state_dict"])
    else:
        ocr_rec_model.load_state_dict(checkpoint)

    f = open(args.wts_save_path, 'w')
    f.write("{}\n".format(len(ocr_rec_model.state_dict().keys())))
    for k,v in ocr_rec_model.state_dict().items():
        print('key: ', k)
        print('value: ', v.shape)
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")
