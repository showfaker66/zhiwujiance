import numpy as np
import argparse
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from wide_resnet_50 import wide_resnet50_2
import torch
from PIL import Image
from torchvision import datasets
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms as T
import pretreatment as mvtec
from torch.autograd import Variable
from other import cosine_Matrix, calc_dist_matrix, visualize_loc_result, denormalization, test_loc_result
import onnxruntime
import onnx


def parse_args():
    parser = argparse.ArgumentParser('SPADE')
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--save_path", type=str, default="./result")
    return parser.parse_args()


def predicted():

    args = parse_args()
    # device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    # load model
    model = wide_resnet50_2(pretrained=True, progress=True)
    model.to(device)
    model.eval()


    outputs = []
    class_name = 'bottle'
    def hook(module, input, output):
        outputs.append(output)

    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)
    model.avgpool.register_forward_hook(hook)


    test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])
    train_feature_filepath = os.path.join(args.save_path, 'temp', 'train_%s.pth' % class_name)
    with open(train_feature_filepath, 'rb') as f:
        train_outputs = pickle.load(f)

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    Image_transform = T.Compose([T.Resize(256, Image.ANTIALIAS),
                                T.CenterCrop(224),
                                T.ToTensor(),
                                T.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])]
                                )
    image = Image.open('./ceshitupian/002.png').convert('RGB')

    x = Image_transform(image).unsqueeze(0)
    test_img = []
    test_img.extend(x.cpu().detach().numpy())


    with torch.no_grad():
        pred = model(x.to(device))

    for k, v in zip(test_outputs.keys(), outputs):
        test_outputs[k].append(v)
    # 初始化 hook outputs
    outputs = []
    #计算图像级距离矩阵
    print('ok1')
    Image_level_threshold = 4.1605787
    test_outputs_avg=torch.cat(tuple(test_outputs['avgpool']), 0)

    dist_matrix = calc_dist_matrix(torch.flatten(test_outputs_avg, 1),    # (1, 2048, 1, 1)
                                   torch.flatten(train_outputs['avgpool'], 1))  # (209, 2048, 1, 1)

    topk_values, topk_indexes = torch.topk(dist_matrix, k=args.top_k, dim=1, largest=False)
    # 8.1367 8.2490 8.2909 8.3552 8.6407   127 140 110 193 68
    scores = torch.mean(topk_values, 1).cpu().detach().numpy() # 8.334506
    if scores <= Image_level_threshold:
        print("无异常")
    else:
        print('存在异常')

    score_map_list = []
    for layer_name in ['layer3']:  # for each layer
        score_maps = []
        # 在K个最近邻的所有像素点上构造一个特征库
        topk_feat_map = train_outputs[layer_name][topk_indexes].squeeze(0) # (5, 1024, 14, 14)
        test_feat_map = torch.cat(tuple(test_outputs[layer_name]), 0) # (1, 1024, 14, 14)
        feat_gallery = topk_feat_map.transpose(3, 1).flatten(0, 2).unsqueeze(-1).unsqueeze(-1) # (980, 1024, 1, 1)
        # 计算距离矩阵
        feat_gallery = feat_gallery.cuda()
        dist_matrix_list = []
        for d_idx in range(feat_gallery.shape[0] // 100):
            dist_matrix = torch.pairwise_distance(feat_gallery[d_idx * 100:d_idx * 100 + 100], test_feat_map)
            dist_matrix_list.append(dist_matrix)
        dist_matrix = torch.cat(dist_matrix_list, 0)
        # 从图库中最接近的K个特征
        score_map = torch.min(dist_matrix, dim=0)[0]
        score_map = F.interpolate(score_map.unsqueeze(0).unsqueeze(0), size=224,
                                  mode='bilinear', align_corners=False)
        score_maps.append(score_map)

    # 取特征间的平均距离
    score_map = torch.mean(torch.cat(score_maps, 0), dim=0)

    # 在异常分数图上使用高斯模糊
    score_map = gaussian_filter(score_map.squeeze().cpu().detach().numpy(), sigma=4)
    score_map_list.append(score_map)
    print(np.array(score_map_list).shape)
    Pixel_level_threshold = [2.2429879]

    # 可视化定位效果
    test_loc_result(test_img, score_map_list, Pixel_level_threshold, save_path='./result/images', class_name=class_name)



    # dummy = Variable(torch.randn(1, 3, 224, 224))
    # torch.onnx.export(model, dummy, 'model.onnx', verbose=True)
    # onnx_model = onnx.load('model.onnx')
    #
    # onnx.checker.check_model(onnx_model)



if __name__ == '__main__':
    predicted()






