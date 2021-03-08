import numpy as np
import time
import cv2
import torch
from torch.autograd import Variable
import lib.utils.utils as utils
import lib.models.crnn as crnn
import lib.config.alphabets as alphabets
import yaml
from easydict import EasyDict as edict
import argparse
import os
 
def parse_arg():
    parser = argparse.ArgumentParser(description="demo")
    parser.add_argument('--test_path', type=str, default='images/test_cn/test.txt', help='the path to your image')
    parser.add_argument('--have_label', action='store_true', default=True, help='if test data have label to cal acc')
    parser.add_argument('--have_underline', action='store_true', default=False, help='if test data have label with 下划线')
    parser.add_argument('--save_root', type=str, default='./output',
                        help='the path to your checkpoints')

    # number
    # parser.add_argument('--cfg', help='experiment configuration filename', type=str, default='lib/config/number_config.yaml')
    # parser.add_argument('--checkpoint', type=str, default='weights/crnn_number.pth', help='the path to your checkpoints')

    # cn
    parser.add_argument('--cfg', help='experiment configuration filename', type=str, default='lib/config/cn_config.yaml')
    parser.add_argument('--checkpoint', type=str, default='weights/crnn_cn.pth', help='the path to your checkpoints')

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f)
        config = edict(config)

    if config.DATASET.CHAR_FILE:
        alphabets_char_file = [char.strip() for char in open(config.DATASET.CHAR_FILE).readlines()[1:]]
        alphabets_char_file = ''.join(alphabets_char_file)
        config.DATASET.ALPHABETS = alphabets_char_file
    else:
        config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args

def recognition(config, img, model, converter, device):

    # github issues: https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec/issues/211
    h, w = img.shape
    # fisrt step: resize the height and width of image to (32, x)
    # img = cv2.resize(img, (0, 0), fx=config.MODEL.IMAGE_SIZE.H / h, fy=config.MODEL.IMAGE_SIZE.H / h, interpolation=cv2.INTER_CUBIC)

    # # old preprocess, not by pad but by resize
    # img = cv2.resize(img, (0, 0), fx=config.MODEL.IMAGE_SIZE.W / w, fy=config.MODEL.IMAGE_SIZE.H / h, interpolation=cv2.INTER_CUBIC)
    # # second step: keep the ratio of image's text same with training
    # h, w = img.shape
    # w_cur = int(img.shape[1] / (config.MODEL.IMAGE_SIZE.OW / config.MODEL.IMAGE_SIZE.W))
    # img = cv2.resize(img, (0, 0), fx=w_cur / w, fy=1.0, interpolation=cv2.INTER_CUBIC)
    # img = np.reshape(img, (config.MODEL.IMAGE_SIZE.H, w_cur, 1))

    ## pad preprocess
    img = utils.gray_img_pad_resize(img, config.MODEL.IMAGE_SIZE.H, config.MODEL.IMAGE_SIZE.W)
    img = np.reshape(img, (config.MODEL.IMAGE_SIZE.H, -1, 1))

    # normalize
    img = img.astype(np.float32)
    img = (img / 255. - config.DATASET.MEAN) / config.DATASET.STD
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    model.eval()
    preds = model(img)
    # print(preds.shape)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    # print('results: {0}'.format(sim_pred))
    return sim_pred

if __name__ == '__main__':

    config, args = parse_arg()
    started = time.time()

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model = crnn.get_crnn(config).to(device)
    print('loading pretrained model from {0}'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)


    image_num = 0
    save_root = args.save_root
    error_count = 0
    error_dir_count = 0
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    image_paths = []
    if os.path.isfile(args.test_path):
        if '.txt' in args.test_path:  # 是数据集list
            with open(args.test_path, 'r') as r:
                lines = r.readlines()
                image_num = len(lines)
                for line in lines:
                    image_path = line.strip().split(' ')[0]
                    image_paths.append(image_path)
        elif os.path.splitext(args.test_path)[-1] in ['.jpg', '.png', '.jpeg']:  # 是图像
            image_paths.append(args.test_path)
        else:
            raise NotImplementedError
    elif os.path.isdir(args.test_path):
        image_num = len(os.listdir(args.test_path))
        for img_name in sorted(os.listdir(args.test_path)):
            image_path = os.path.join(args.test_path, img_name)
            image_paths.append(image_path)
    else: raise NotImplementedError("input path problem")

    for image_path in image_paths:
        img_src = cv2.imread(image_path)
        try:
            img = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
        except:
            print("%s error, maybe not found" % image_path)
            continue
        converter = utils.strLabelConverter(config.DATASET.ALPHABETS)

        # cv2.imwrite("/mnt/data1/temp_out/test.jpg", img)
        rec_res = recognition(config, img, model, converter, device)
        if '.png' in image_path: suffix = '.png'
        elif '.jpg' in image_path: suffix = '.jpg'
        else: raise NotImplementedError
        image_name = image_path.split('/')[-1].split(suffix)[0]
        if args.have_label:
            save_name = rec_res + '_' + image_name + suffix  # 错误label放在第一位
            if not args.have_underline:
                image_label = image_name.split('_')[-1]
            else:  # label中存在下划线的情况,默认这种数据第一个下划线后的字符都是label
                image_label = '_'.join(image_name.split('_')[1:]) 
            if rec_res != image_label:
                error_count += 1
                if error_count % 5000 == 0:
                    error_dir_count += 1
                save_dir_path = os.path.join(save_root, str(error_dir_count))
                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path)
                cv2.imwrite(os.path.join(save_dir_path, save_name), img_src)
        else:  # 正常前向
            save_name =  image_name + '_' + rec_res + suffix  # 错误label放在第一位
            cv2.imwrite(os.path.join(save_root, save_name), img_src)

    if args.have_label:
        print("[#correct:{} / #total:{}]".format(image_num-error_count, image_num))
        print("acc: %f" % ((image_num-error_count)/image_num))

    finished = time.time()
    print('elapsed time: {0}'.format(finished - started))

