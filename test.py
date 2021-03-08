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
from lib.dataset import get_dataset
from lib.core.function import AverageMeter
from torch.utils.data import DataLoader


def parse_arg():
    parser = argparse.ArgumentParser(description="demo")

    parser.add_argument('--cfg', help='experiment configuration filename', type=str, default='lib/config/CF_config.yaml')
    parser.add_argument('--test_path', type=str, default='/mnt/data1/number/gen10w/datalist_val.txt', help='the path to your image')
    parser.add_argument('--checkpoint', type=str, default='/home/ps/yangguang/opensource_lib/CRNN_Chinese_Characters_Rec/output/CF/crnn_2char_big_number_trainval_second_try/2020-11-23-10-45/checkpoints/checkpoint_99_acc_0.9848.pth',
                        help='the path to your checkpoints')

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

    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args


def validate(config, val_loader, dataset, converter, model, criterion, device):

    losses = AverageMeter()
    model.eval()
    n_correct = 0
    with torch.no_grad():
        for i, (inp, idx) in enumerate(val_loader):
            labels = utils.get_batch_label(dataset, idx)
            inp = inp.to(device)
            # inference
            preds = model(inp).cpu()

            # compute loss
            batch_size = inp.size(0)
            text, length = converter.encode(labels)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)
            loss = criterion(preds, text, preds_size, length)

            losses.update(loss.item(), inp.size(0))

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            for pred, target in zip(sim_preds, labels):
                if pred == target:
                    n_correct += 1

            if i == config.TEST.NUM_TEST_BATCH:
                break

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:config.TEST.NUM_TEST_DISP]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, labels):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    num_test_sample = config.TEST.NUM_TEST_BATCH * config.TEST.BATCH_SIZE_PER_GPU
    if num_test_sample > len(dataset):
        num_test_sample = len(dataset)

    print("[#correct:{} / #total:{}]".format(n_correct, num_test_sample))
    accuracy = n_correct / float(num_test_sample)
    print('Test loss: {:.4f}, accuray: {:.4f}'.format(losses.avg, accuracy))

    return accuracy


def main():
    config, args = parse_arg()
    model = crnn.get_crnn(config)
    criterion = torch.nn.CTCLoss()
    # get device
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(config.GPUID))
    else:
        device = torch.device("cpu:0")
    model = model.to(device)
    val_dataset = get_dataset(config)(config, is_train=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
    acc = validate(config, val_loader, val_dataset, converter, model, criterion, device)


if __name__ == "__main__":
    main()