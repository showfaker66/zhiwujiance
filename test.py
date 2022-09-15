import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2
import pretreatment as mvtec
from other import cosine_Matrix, calc_dist_matrix, visualize_loc_result, denormalization
def parse_args():
    parser = argparse.ArgumentParser('SPADE')
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--save_path", type=str, default="./result")
    return parser.parse_args()

def main():

    args = parse_args()

    # device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    # load model
    model = wide_resnet50_2(pretrained=True, progress=True)
    model.to(device)
    model.eval()

    # 选取特征块
    outputs = []
    def hook(module, input, output):
        outputs.append(output)
    model.layer1[-1].register_forward_hook(hook)   #抽取56*56*256中的最后一个56*56的特征向量
    model.layer2[-1].register_forward_hook(hook)   #抽取28*28*512中的最后一个28*28的特征向量
    model.layer3[-1].register_forward_hook(hook)   #抽取14*14*1024中的最后一个14*14的特征向量
    model.avgpool.register_forward_hook(hook)      #抽取最后的1*1*1000的最后一个全连接层的特征向量

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    total_roc_auc = []
    total_pixel_roc_auc = []
    class_name = 'bottle'
    train_feature_filepath = os.path.join(args.save_path, 'temp', 'train_%s.pth' % class_name)
    with open(train_feature_filepath, 'rb') as f:
        train_outputs = pickle.load(f)
    test_dataset = mvtec.MVTecDataset(class_name=class_name, is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)
    test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])

    gt_list = []
    gt_mask_list = []
    test_imgs = []

    # 提取测试集特征 x: (22, 3, 224, 224)  y: (1, 22)  mask: (22, 1, 224, 224)
    for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):
        test_imgs.extend(x.cpu().detach().numpy())
        gt_list.extend(y.cpu().detach().numpy())
        gt_mask_list.extend(mask.cpu().detach().numpy())
        # model prediction
        with torch.no_grad():
            pred = model(x.to(device))  # (22, 1000)
        # 获取中间层
        for k, v in zip(test_outputs.keys(), outputs):
            test_outputs[k].append(v)
        # 初始化 hook outputs
        outputs = []
    for k, v in test_outputs.items():
        test_outputs[k] = torch.cat(v, 0)  # (83, 2048, 1, 1)

    # 计算距离矩阵
    dist_matrix = calc_dist_matrix(torch.flatten(test_outputs['avgpool'], 1),  # (83, 2048)
                               torch.flatten(train_outputs['avgpool'], 1))  # (209, 2048)
    # print(np.array(dist_matrix).shape) #(83, 209) -> (22, 209)  22张测试图片，每张测试和209张训练图片的距离值
    # dist_matrix是尺寸为(83,209)的矩阵，每一个83维中包含了单个测试图像对每张训练图像的距离值（训练图像有209个）

    # 选择K个最近邻并取平均值
    topk_values, topk_indexes = torch.topk(dist_matrix, k=args.top_k, dim=1, largest=False)
    # 选出距离最小的5个
    # print(np.array(topk_values).shape) (83, 5)
    # print(np.array(topk_indexes).shape) (83, 5)
    scores = torch.mean(topk_values, 1).cpu().detach().numpy()

    # 计算 image-level ROC——AUC 分数
    fpr, tpr, _ = roc_curve(gt_list, scores)
    print('image_level_threshold:', _[-1])
    roc_auc = roc_auc_score(gt_list, scores)
    # print('roc_auc', roc_auc)
    total_roc_auc.append(roc_auc)
    # print('%s ROCAUC: %.3f' % (class_name, roc_auc))
    fig_img_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, roc_auc))

    score_map_list = []
    for t_idx in tqdm(range(test_outputs['avgpool'].shape[0]), '| localization | test | %s |' % class_name):
        score_maps = []
        for layer_name in ['layer3']:  # for each layer
            # 在K个最近邻的所有像素点上构造一个特征库
            topk_feat_map = train_outputs[layer_name][topk_indexes[t_idx]]
            # print('topk_feat_map', np.array(topk_feat_map).shape)  #(5, 256, 56, 56)
            test_feat_map = test_outputs[layer_name][t_idx:t_idx + 1]
            # print('test_feat_map:', np.array(test_feat_map).shape) #(1, 256, 56, 56)
            feat_gallery = topk_feat_map.transpose(3, 1).flatten(0, 2).unsqueeze(-1).unsqueeze(-1)
            # print('feat_gallery.shape:', np.array(feat_gallery).shape)  #(15680, 256, 1, 1)
            feat_gallery = feat_gallery.cuda()

            # 计算距离矩阵
            dist_matrix_list = []
            for d_idx in range(feat_gallery.shape[0] // 100):  # 156
                dist_matrix = torch.pairwise_distance(feat_gallery[d_idx * 100:d_idx * 100 + 100], test_feat_map)
                # print('test_feat_map:', np.array(test_feat_map).shape)
                # print('feat:', np.array(feat_gallery[d_idx * 100:d_idx * 100 + 100]).shape) #(100, 256, 1, 1)
                # print('dist_matrix:', np.array(dist_matrix.shape)) #(100, 56, 56)
                dist_matrix_list.append(dist_matrix)

            dist_matrix = torch.cat(dist_matrix_list, 0)
            # print(np.array(dist_matrix).shape)    #(15600, 56, 56)
            # 从图库中最接近的K个特征
            score_map = torch.min(dist_matrix, dim=0)[0]
            # print(np.array(score_map).shape) #(56, 56)
            score_map = F.interpolate(score_map.unsqueeze(0).unsqueeze(0), size=224,
                                    mode='bilinear', align_corners=False)
            # print('score_map_F:', np.array(score_map).shape) #(1, 1, 224, 224)
            score_maps.append(score_map)

        # 取特征间的平均距离
        score_map = torch.mean(torch.cat(score_maps, 0), dim=0)

        # 在异常分数图上使用高斯模糊
        score_map = gaussian_filter(score_map.squeeze().cpu().detach().numpy(), sigma=4)
        score_map_list.append(score_map)
        print(np.array(score_map_list).shape)

    flatten_gt_mask_list = np.concatenate(gt_mask_list).ravel()
    flatten_score_map_list = np.concatenate(score_map_list).ravel()

    # 计算每个像素的ROC--AUC分数
    fpr, tpr, _ = roc_curve(flatten_gt_mask_list, flatten_score_map_list)
    per_pixel_rocauc = roc_auc_score(flatten_gt_mask_list, flatten_score_map_list)
    total_pixel_roc_auc.append(per_pixel_rocauc)
    print('%s pixel ROCAUC: %.3f' % (class_name, per_pixel_rocauc))
    fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))

    # 获取最优阈值
    precision, recall, thresholds = precision_recall_curve(flatten_gt_mask_list, flatten_score_map_list)
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]
    print('pixel_level_threshold:', threshold)

    # 可视化定位效果
    visualize_loc_result(test_imgs, gt_mask_list, score_map_list, threshold, args.save_path, class_name, vis_num=5)

    print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.legend(loc="lower right")

    print('Average pixel ROCUAC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(args.save_path, 'roc_curve.png'), dpi=100)

if __name__ == '__main__':
    main()