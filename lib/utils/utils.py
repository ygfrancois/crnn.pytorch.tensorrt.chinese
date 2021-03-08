import torch.optim as optim
import time
from pathlib import Path
import os
import torch
import random
import cv2
import numpy as np
import cv2


def get_optimizer(config, model):

    optimizer = None

    if config.TRAIN.OPTIMIZER == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.TRAIN.LR,
            momentum=config.TRAIN.MOMENTUM,
            weight_decay=config.TRAIN.WD,
            nesterov=config.TRAIN.NESTEROV
        )
    elif config.TRAIN.OPTIMIZER == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.TRAIN.LR,
        )
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
    elif config.TRAIN.OPTIMIZER == "rmsprop":
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.TRAIN.LR,
            momentum=config.TRAIN.MOMENTUM,
            weight_decay=config.TRAIN.WD,
            # alpha=config.TRAIN.RMSPROP_ALPHA,
            # centered=config.TRAIN.RMSPROP_CENTERED
        )

    return optimizer

def create_log_folder(cfg, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    checkpoints_output_dir = root_output_dir / dataset / model / time_str / 'checkpoints'

    print('=> creating {}'.format(checkpoints_output_dir))
    checkpoints_output_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_log_dir = root_output_dir / dataset / model / time_str / 'log'
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)


    return {'chs_dir': str(checkpoints_output_dir), 'tb_dir': str(tensorboard_log_dir)}


def get_batch_label(d, i):
    label = []
    for idx in i:
        label.append(list(d.labels[idx].values())[0])
    return label

class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """

        length = []
        result = []
        decode_flag = True if type(text[0])==bytes else False

        for item in text:

            if decode_flag:
                item = item.decode('utf-8','strict')
            length.append(len(item))
            for char in item:
                try:
                    index = self.dict[char]
                except:
                    import pdb;pdb.set_trace()
                    print(char)
                result.append(index)
        text = result
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts

    def decode_np(self, preds, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if len(preds.shape) == 1:
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(preds.shape[0]):
                    if preds[i] != 0 and (not (i > 0 and preds[i - 1] == preds[i])):
                        char_list.append(self.alphabet[preds[i] - 1])
                return ''.join(char_list)
        else: 
            raise NotImplementedError
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts

def get_char_dict(path):
    with open(path, 'rb') as file:
        char_dict = {num: char.strip().decode('gbk', 'ignore') for num, char in enumerate(file.readlines())}

def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))


### 数据拓增
# gauss blur
def apply_gauss_blur(img, ks=None):
    if ks is None:
        ks = [7, 9, 11, 13]
    ksize = random.choice(ks)

    sigmas = [0, 1, 2, 3, 4, 5, 6, 7]
    sigma = 0
    if ksize <= 3:
        sigma = random.choice(sigmas)
    img = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    return img


# norm_blur
def apply_norm_blur(img, ks=None):
    # kernel == 1, the output image will be the same
    if ks is None:
        ks = [2, 3]
    kernel = random.choice(ks)
    img = cv2.blur(img, (kernel, kernel))
    return img


# 颜色取倒
def reverse_img(word_img):
    offset = np.random.randint(-10, 10)
    return 255 + offset - word_img


# 颜色通道翻转
def reverse_img_color_channel(img):
    img = img[:,:, ::-1]
    return img


# 滤波
def apply_emboss(word_img):
    emboss_kernal = np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
    ])
    return cv2.filter2D(word_img, -1, emboss_kernal)

def apply_sharp(word_img):
    sharp_kernel = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ])
    return cv2.filter2D(word_img, -1, sharp_kernel)

# RGB转换到HSV空间后进行HSV空间的扰动，再转换回RGB
def random_distort(img, hue=0.5, saturation=0.5, exposure=0.5):
    """
    perform random distortion in the HSV color space.
    Args:
        img (numpy.ndarray): input image whose shape is :math:`(H, W, C)`.
            Values range from 0 to 255.
        hue (float): random distortion parameter.
        saturation (float): random distortion parameter.
        exposure (float): random distortion parameter.
    Returns:
        img (numpy.ndarray)
    """
    def rand_scale(s):
        #乘或者除一定倍数
        """
        calculate 
        random scaling factor
        Args:
            s (float): range of the random scale.
        Returns:
            random scaling factor (float) whose range is
            from 1 / s to s .
        """
        scale = np.random.uniform(low=1, high=s)
        if np.random.rand() > 0.5:
            return scale
        return 1 / scale
    #hue 调整色彩度，越大色彩度变化的程度越大；sat 调整对比度，越大对比度变化越大； exp调整亮度
    dhue = np.random.uniform(low=-hue, high=hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
 
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = np.asarray(img, dtype=np.float32) / 255.
    img[:, :, 1] *= dsat
    img[:, :, 2] *= dexp
    H = img[:, :, 0] + dhue
 
    if dhue > 0:
        H[H > 1.0] -= 1.0
    else:
        H[H < 0.0] += 1.0
 
    img[:, :, 0] = H
    img = (img * 255).clip(0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    img = np.asarray(img, dtype=np.float32)
 
    return img

def change_image_resolution_by_resize(img, ratio=0.5):
    try:
        h,w = img.shape[:2]
        img = cv2.resize(img, (int(w*ratio),int(h*ratio)))
        img = cv2.resize(img, (w, h))
    except:
        print("!")
    return img


def change_image_brightness(img_bgr, ratio=0.5):
    try:
        img = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        img_hsv[:,:,2] = img_hsv[:,:,2] * ratio
    except:
        print("!")
    img_out = img_hsv/255.0
    mask_1 = img_out  < 0 
    mask_2 = img_out  > 1
    img_out = img_out * (1-mask_1)
    img_out = img_out * (1-mask_2) + mask_2
    img_out = img_out * 255.0
    # HSV转RGB
    img_out = np.round(img_out).astype(np.uint8)
    img_out = cv2.cvtColor(img_out, cv2.COLOR_HSV2RGB)
    img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)


    return img_out


def random_img_aug(img, aug_kinds=1):
    aug_types = random.sample([1,2,3,4,5,6, 7], aug_kinds)
    # aug_types = random.sample([1,2,3,4,5,6,7,8], aug_kinds)
    for aug_type in aug_types:
        if aug_type ==1:
            img = apply_gauss_blur(img)
        elif aug_type == 2:
            img = apply_norm_blur(img)
        elif aug_type == 3:
            img = reverse_img(img)
        elif aug_type == 4:
            img= reverse_img_color_channel(img)
        elif aug_type == 5:
            img = apply_emboss(img)
        elif aug_type == 6:
            img = apply_sharp(img)
        # elif aug_type == 7:
        #     img = random_distort(img)
        elif aug_type == 7:
            img = change_image_resolution_by_resize(img, ratio=random.uniform(0.6,0.9))
        elif aug_type == 8:
            img = change_image_brightness(img, ratio=random.uniform(0.6,0.9))
        else: NotImplementedError
    return img

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


if __name__ == "__main__":
    img = cv2.imread("/mnt/data1/CF/Gen_by_YG/cf_back3/20/text_images/41/00099723_0_窘B粱邾摁.jpg",0)
    cv2.imwrite("/mnt/data1/temp_out/temp.png", img)

    img_pad_resize = gray_img_pad_resize(img, 32, 160)

    cv2.imwrite("/mnt/data1/temp_out/temp1.png", img_pad_resize)