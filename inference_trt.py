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

# tensorrt
import ctypes
import random
import sys
import threading
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torchvision

 
def parse_arg():
    parser = argparse.ArgumentParser(description="demo")
    parser.add_argument('--test_path', type=str, default='images/test_cn/test.txt', help='the path to your image')
    parser.add_argument('--have_label', action='store_true', default=True, help='if test data have label to cal acc')
    parser.add_argument('--have_underline', action='store_true', default=False, help='if test data have label with underline')
    parser.add_argument('--save_root', type=str, default='./output',
                        help='the path to your checkpoints')

    # number
    # parser.add_argument('--cfg', help='experiment configuration filename', type=str, default='lib/config/number_config.yaml')
    # parser.add_argument('--engine_file_path', type=str, default='"weights/crnn_number.engine"', help='the path to your checkpoints')

    # cn
    parser.add_argument('--cfg', help='experiment configuration filename', type=str, default='lib/config/cn_config.yaml')
    parser.add_argument('--engine_file_path', type=str, default='weights/crnn_cn.engine', help='the path to your checkpoints')
                        
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


class CRNNTRT(object):
    """
    description: A CRNN class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path, num_class):
        # Create a Context on this device,
        self.cfx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.num_class = num_class

    def infer(self, input_image_path):
        threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.cfx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess
        input_image, image_src = self.preprocess_image(input_image_path)
        # Copy input image to host buffer
        np.copyto(host_inputs[0], input_image.ravel())
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]

        preds = np.argmax(np.reshape(output,(-1,self.num_class)), axis=1) 
        sim_pred = converter.decode_np(preds, raw=False)

        return sim_pred, image_src

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()

    def preprocess_image(self, input_image_path):
        """
        description: Read an image from image path, convert it to RGB,
                     resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        img_src = cv2.imread(input_image_path)
        try:
            img = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
        except:
            print("%s error, maybe not found" % input_image_path)
            return
        
        h, w = img.shape
        
        ## pad preprocess
        img = utils.gray_img_pad_resize(img, config.MODEL.IMAGE_SIZE.H, config.MODEL.IMAGE_SIZE.W)
        img = np.reshape(img, (config.MODEL.IMAGE_SIZE.H, -1, 1))

        # normalize
        img = img.astype(np.float32)
        img = (img / 255. - config.DATASET.MEAN) / config.DATASET.STD
        img = img.transpose([2, 0, 1])
        img = np.expand_dims(img, axis=0)

        return img, img_src


if __name__ == '__main__':
    config, args = parse_arg()
    started = time.time()

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

    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
    crnn_wrapper = CRNNTRT(args.engine_file_path, config.MODEL.NUM_CLASSES+1)

    for image_path in image_paths:
        # cv2.imwrite("/mnt/data1/temp_out/test.jpg", img)
        rec_res, img_src = crnn_wrapper.infer(image_path)
        # rec_res = recognition(config, img, model, converter, device)
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
    
    crnn_wrapper.destroy()

    if args.have_label:
        print("[#correct:{} / #total:{}]".format(image_num-error_count, image_num))
        print("acc: %f" % ((image_num-error_count)/image_num))

    finished = time.time()
    print('elapsed time: {0}'.format(finished - started))

