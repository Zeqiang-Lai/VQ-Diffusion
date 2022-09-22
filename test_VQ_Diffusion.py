# ------------------------------------------
# VQ-Diffusion
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Shuyang Gu
# ------------------------------------------

from image_synthesis.utils.misc import get_model_parameters_info
from image_synthesis.modeling.build import build_model
from image_synthesis.utils.io import load_yaml_config
from PIL import Image
import torchvision
import numpy as np
import argparse
import cv2
import torch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


class VQ_Diffusion():
    def __init__(self, config, path):
        self.info = self.get_model(ema=True, model_path=path, config_path=config)
        self.model = self.info['model']
        self.epoch = self.info['epoch']
        self.model_name = self.info['model_name']
        self.model = self.model.cuda()
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def get_model(self, ema, model_path, config_path):
        if 'OUTPUT' in model_path:  # pretrained model
            model_name = model_path.split(os.path.sep)[-3]
        else:
            model_name = os.path.basename(config_path).replace('.yaml', '')

        config = load_yaml_config(config_path)
        model = build_model(config)
        model_parameters = get_model_parameters_info(model)

        print(model_parameters)
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location="cpu")

        if 'last_epoch' in ckpt:
            epoch = ckpt['last_epoch']
        elif 'epoch' in ckpt:
            epoch = ckpt['epoch']
        else:
            epoch = 0

        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print('Model missing keys:\n', missing)
        print('Model unexpected keys:\n', unexpected)

        if ema == True and 'ema' in ckpt:
            print("Evaluate EMA model")
            ema_model = model.get_ema_model()
            missing, unexpected = ema_model.load_state_dict(ckpt['ema'], strict=False)

        return {'model': model, 'epoch': epoch, 'model_name': model_name, 'parameter': model_parameters}

    def inference_generate_sample_with_class(self, text, truncation_rate, save_root, batch_size, fast=False):
        os.makedirs(save_root, exist_ok=True)

        data_i = {}
        data_i['label'] = [text]
        data_i['image'] = None
        condition = text

        str_cond = str(condition)
        save_root_ = os.path.join(save_root, str_cond)
        os.makedirs(save_root_, exist_ok=True)

        with torch.no_grad():
            model_out = self.model.generate_content(
                batch=data_i,
                filter_ratio=0,
                replicate=batch_size,
                content_ratio=1,
                return_att_weight=False,
                sample_type="top"+str(truncation_rate)+'r',
            )  # B x C x H x W

        # save results
        content = model_out['content']
        content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
        for b in range(content.shape[0]):
            cnt = b
            save_base_name = '{}'.format(str(cnt).zfill(6))
            save_path = os.path.join(save_root_, save_base_name+'.jpg')
            im = Image.fromarray(content[b])
            im.save(save_path)

    def inference_generate_sample_with_condition(self, texts, truncation_rate, save_root, batch_size, fast=False):
        os.makedirs(save_root, exist_ok=True)

        data_i = {}
        data_i['text'] = texts
        data_i['image'] = None

        if fast != False:
            add_string = 'r,fast'+str(fast-1)
        else:
            add_string = 'r'
        with torch.no_grad():
            model_out = self.model.generate_content(
                batch=data_i,
                filter_ratio=0,
                replicate=1,
                content_ratio=1,
                return_att_weight=False,
                sample_type="top"+str(truncation_rate)+add_string,
            )  # B x C x H x W

        # save results
        content = model_out['content']
        content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
        for b in range(content.shape[0]):
            from datetime import datetime
            dtstr = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            save_base_name = f'{dtstr}_{b}'
            save_path = os.path.join(save_root, save_base_name+'.png')
            im = Image.fromarray(content[b])
            im.save(save_path)


class CUB200TestDatast(torch.utils.data.Dataset):
    def __init__(self, caption_path):
        super().__init__()
        with open(caption_path, 'r') as f:
            captions = f.readlines()
            self.captions = [caption.strip() for caption in captions]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        return self.captions[idx]


if __name__ == '__main__':
    from tqdm import tqdm

    with open('cub_test_captions.txt', 'r') as f:
        captions = f.readlines()
        captions = [caption.strip() for caption in captions]

    VQ_Diffusion = VQ_Diffusion(
        config='OUTPUT/pretrained_model/config_text.yaml',
        path='OUTPUT/pretrained_model/cub_pretrained.pth'
    )

    bs = 8
    batches = [captions[i:i+bs] for i in range(0, len(captions), bs)]
    for batch in tqdm(batches):
        VQ_Diffusion.inference_generate_sample_with_condition(
            batch,
            truncation_rate=0.86,
            save_root="TEST/cub_pretrained_30k",
            batch_size=8,
        )
