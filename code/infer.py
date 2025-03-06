import glob
import sys
from collections import OrderedDict
from tqdm import tqdm
from natsort import natsort
import argparse
import options.options as option
from models import create_model
import torch
from utils.util import opt_get
import numpy as np
import pandas as pd
import os
import cv2
import warnings

warnings.filterwarnings("ignore")


def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))


def load_model(conf_path):
    opt = option.parse(conf_path, is_train=False)
    opt['gpu_ids'] = None
    opt = option.dict_to_nonedict(opt)
    model = create_model(opt)

    model_path = opt_get(opt, ['model_path'], None)
    model.load_network(load_path=model_path, network=model.netG)
    return model, opt


def predict(model, lr):
    model.feed_data({"LQ": t(lr)}, need_GT=False)
    model.test()
    visuals = model.get_current_visuals(need_GT=False)
    return visuals.get('rlt', visuals.get('NORMAL'))


def t(array):
    return torch.Tensor(np.expand_dims(array.transpose([2, 0, 1]), axis=0).astype(np.float32)) / 255


def rgb(t):
    return (np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(np.uint8)


def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]


def imwrite(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img[:, :, [2, 1, 0]])


def auto_padding(img, times=16):
    h, w, _ = img.shape
    h1, w1 = (times - h % times) // 2, (times - w % times) // 2
    h2, w2 = (times - h % times) - h1, (times - w % times) - w1
    img = cv2.copyMakeBorder(img, h1, h2, w1, w2, cv2.BORDER_REFLECT)
    return img, [h1, h2, w1, w2]


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--opt", default="./code/confs/LOL.yml")
    parser.add_argument("--opt", default="./code/confs/LOL-v2-real.yml")
    parser.add_argument("-n", "--name", default="unpaired")
    args = parser.parse_args()
    conf_path = args.opt
    conf = conf_path.split('/')[-1].replace('.yml', '')
    model, opt = load_model(conf_path)
    device = 'cuda:0'
    model.netG = model.netG.to(device)
    model.net_hq = model.net_hq.to(device)

    lr_dir = opt['dataroot_unpaired']
    lr_paths = fiFindByWildcard(os.path.join(lr_dir, '*.*'))

    # test_dir = 'DarkFace_Train_2021/GLARE-test'
    test_dir = 'DarkFace_Train_2021/GLARE-LOL-v2-real'
    os.makedirs(test_dir, exist_ok=True)
    print(f"Out dir: {test_dir}")

    for lr_path, idx_test in zip(tqdm(lr_paths), range(len(lr_paths))):
    # for lr_path, idx_test in zip(lr_paths, range(len(lr_paths))):
        path_out_sr = os.path.join(test_dir, os.path.basename(lr_path))
        if not os.path.exists(path_out_sr):
        # if os.path.basename(lr_path) in ['1.png', '80.png', '89.png', '134.png', '184.png', '755.png']:
            lr = imread(lr_path)
            raw_shape = lr.shape

            # 패딩 추가
            lr, padding_params = auto_padding(lr)

            lr_t = t(lr)
            if opt["datasets"]["train"].get("log_low", False):
                lr_t = torch.log(torch.clamp(lr_t + 1e-3, min=1e-3))
                # print(f"Log applied to lr_t: min={lr_t.min().item()}, max={lr_t.max().item()}")

            heat = opt['heat']

            # print(f"[Input] Image: {os.path.basename(lr_path)}, Mean: {lr_t.mean()}, Std: {lr_t.std()}, Min: {lr_t.min()}, Max: {lr_t.max()}")
            with torch.no_grad():
                sr_t = model.get_sr(lq=lr_t.cuda(), heat=None)

            # print(f"[Output] Image: {os.path.basename(lr_path)}, Mean: {sr_t.mean()}, Std: {sr_t.std()}, Min: {sr_t.min()}, Max: {sr_t.max()}")
            # print()
            # print()
            print()

            # 기존 후처리 코드 (클램핑, 패딩 제거, rgb 변환) - 주석 처리
            sr = rgb(torch.clamp(sr_t, 0, 1)[:, :, padding_params[0]:sr_t.shape[2] - padding_params[1],
                        padding_params[2]:sr_t.shape[3] - padding_params[3]])
            assert raw_shape == sr.shape

            imwrite(path_out_sr, sr)

            # ✅ 후처리: Min-Max 정규화 적용 (clamp 제거)
            # sr_t_min, sr_t_max = sr_t.min(), sr_t.max()
            # print(f"sr_t range before normalization: min={sr_t_min.item()}, max={sr_t_max.item()}")

            # if sr_t_max - sr_t_min > 1e-6:  # min-max 정규화 적용 가능할 때만
            #     sr_t = (sr_t - sr_t_min) / (sr_t_max - sr_t_min)
            # else:
            #     print("Warning: sr_t has very small variation, skipping normalization")

            # # 패딩 제거
            # sr_cropped = sr_t[:, :, padding_params[0]:sr_t.shape[2] - padding_params[1], padding_params[2]:sr_t.shape[3] - padding_params[3]]
            # print(f"Cropped sr_t shape: {sr_cropped.shape}")

            # # RGB 변환 후 저장
            # sr = rgb(sr_cropped)
            # print(f"sr range after processing: min={sr.min()}, max={sr.max()}")

            # 크기 검사 후 저장
            # assert raw_shape == sr.shape, f"Mismatch in shape! Expected {raw_shape}, got {sr.shape}"
            # path_out_sr = os.path.join(test_dir, os.path.basename(lr_path))
            # imwrite(path_out_sr, sr)


if __name__ == "__main__":
    main()
