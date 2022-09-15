import numpy as np
import os
import matplotlib.pyplot as plt
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'
#余弦相似距离
def cosine_Matrix(a, b):
    x1, y1, z1 = a.shape
    result_cosine_Matrix = []

    for i in range(x1):
        A = np.matrix(a[i])
        B = np.matrix(b[i])

        one_example = np.divide(np.multiply(A, B).sum(1), np.multiply(np.sqrt(np.multiply(A, A).sum(1)),
                                                      np.sqrt(np.multiply(B, B).sum(1))))

        result_cosine_Matrix.append(one_example)

    result_cosine_Matrix = np.array(result_cosine_Matrix).squeeze(2)
    x2, y2 = result_cosine_Matrix.shape  # (1, 209)
    for i in range(x2):
        for j in range(y2):
            result_cosine_Matrix[i][j] =1 - result_cosine_Matrix[i][j]
    #result = np.ones(x1, y1) - result_cosine_Matrix
    result_cosine_Matrix = torch.from_numpy(result_cosine_Matrix)
    result_cosine_Matrix = result_cosine_Matrix.cuda()
    return 10*result_cosine_Matrix

def calc_dist_matrix(x, y):
    """用torch张量计算欧几里得距离矩阵"""
    n = x.size(0)   #83
    m = y.size(0)   #209
    d = x.size(1)   #2048
    x = x.unsqueeze(1).expand(n, m, d) #x:(83, 2048) x.unsqueeze(1)=(83, 1, 2048) x.expand(n,m,d)=(83, 209, 2048)
    y = y.unsqueeze(0).expand(n, m, d) #y:(209, 2048)y.unsqueeze(0)=(1, 209,2048) y.expand(n,m,d)=(83, 209, 2048)
    # x = x.cuda()
    y = y.cuda()
    dist_matrix = torch.sqrt(torch.pow(x - y, 2).sum(2))

    x = x.cpu()
    y = y.cpu()

    #添加余弦相似距离
    dist_matrix = dist_matrix+cosine_Matrix(x, y)

    return dist_matrix



def visualize_loc_result(test_imgs, gt_mask_list, score_map_list, threshold,
                         save_path, class_name, vis_num=5):

    for t_idx in range(vis_num):
        test_img = test_imgs[t_idx]
        test_img = denormalization(test_img)
        test_gt = gt_mask_list[t_idx].transpose(1, 2, 0).squeeze()
        test_pred = score_map_list[t_idx]
        test_pred[test_pred <= threshold] = 0
        test_pred[test_pred > threshold] = 1
        test_pred_img = test_img.copy()
        test_pred_img[test_pred == 0] = 0

        fig_img, ax_img = plt.subplots(1, 4, figsize=(12, 4))
        fig_img.subplots_adjust(left=0, right=1, bottom=0, top=1)

        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)

        ax_img[0].imshow(test_img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(test_gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax_img[2].imshow(test_pred, cmap='gray')
        ax_img[2].title.set_text('Predicted mask')
        ax_img[3].imshow(test_pred_img)
        ax_img[3].title.set_text('Predicted anomalous image')

        os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
        fig_img.savefig(os.path.join(save_path, 'images', '%s_%03d.png' % (class_name, t_idx)), dpi=100)
        fig_img.clf()
        plt.close(fig_img)



def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x

def test_denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = np.array(x).squeeze(0)
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


def test_loc_result(test_img, score_map_list, threshold,
                    save_path, class_name):


    test_img = test_denormalization(test_img)
    test_pred = score_map_list[0]
    test_pred[test_pred <= threshold] = 0
    test_pred[test_pred > threshold] = 1
    test_pred_img = test_img.copy()
    test_pred_img[test_pred == 0] = 0

    fig_img, ax_img = plt.subplots(1, 3, figsize=(12, 4))
    fig_img.subplots_adjust(left=0, right=1, bottom=0, top=1)

    for ax_i in ax_img:
        ax_i.axes.xaxis.set_visible(False)
        ax_i.axes.yaxis.set_visible(False)

    ax_img[0].imshow(test_img)
    ax_img[0].title.set_text('Image')
    ax_img[1].imshow(test_pred, cmap='gray')
    ax_img[1].title.set_text('Predicted mask')
    ax_img[2].imshow(test_pred_img)
    ax_img[2].title.set_text('Predicted anomalous image')

    os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
    fig_img.savefig(os.path.join(save_path, 'images', '%s03d.png' % (class_name)), dpi=100)
    fig_img.clf()
    plt.close(fig_img)