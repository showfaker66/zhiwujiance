import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2
import pretreatment as mvtec


def parse_args():
    parser = argparse.ArgumentParser('SPADE')
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--save_path", type=str, default="./result")
    return parser.parse_args()

def main():

    args = parse_args()

    # device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model
    model = wide_resnet50_2(pretrained=True, progress=True)
    model.to(device)
    model.eval()

    outputs = []
    def hook(module, input, output):
        outputs.append(output)
    model.layer1[-1].register_forward_hook(hook)   #抽取56*56*256中的最后一个56*56的特征向量
    model.layer2[-1].register_forward_hook(hook)   #抽取28*28*512中的最后一个28*28的特征向量
    model.layer3[-1].register_forward_hook(hook)   #抽取14*14*1024中的最后一个14*14的特征向量
    #model.layer4[-1].register_forward_hook(hook)
    model.avgpool.register_forward_hook(hook)      #抽取最后的1*1*2048的最后一个全连接层的特征向量

    os.makedirs(os.path.join(args.save_path, 'temp'), exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    '''
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    total_roc_auc = []
    total_pixel_roc_auc = []
    '''

    train_dataset = mvtec.MVTecDataset(class_name='bottle', is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)

    train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])
    # 获取训练集特征
    train_feature_filepath = os.path.join(args.save_path, 'temp', 'train_bottle.pth')
    for (x, y, mask) in tqdm(train_dataloader, '| feature extraction | train | bottles |'):
        print(x.shape)
        with torch.no_grad():
            pred = model(x.to(device))
            # 获取所需特征块
        for k, v in zip(train_outputs.keys(), outputs):
            train_outputs[k].append(v)
            # 初始化output
        outputs = []
    for k, v in train_outputs.items():
        train_outputs[k] = torch.cat(v, 0)
    # 保存训练集特征
    with open(train_feature_filepath, 'wb') as f:
        print(f)
        pickle.dump(train_outputs, f)


if __name__ == '__main__':
    main()