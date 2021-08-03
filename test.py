from model import cfcn
import torch
import PIL.Image as Image
import numpy as np
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from util import non_max_suppression, get_metrics, DetDataset, ClsDataset
from scipy.io import loadmat
import cv2
import torch.nn.functional as F
import h5py

matplotlib.use('Agg')
epsilon = 1e-7


def draw_gt(mat, _img, limit):
    img = _img.copy()
    for i in range(mat.shape[0]):
        y = mat[i][0]
        x = mat[i][1]
        if limit is None:
            img = cv2.circle(img, center=(y, x), radius=5, color=1, thickness=1)
        else:
            if np.round(x) >= 500:
                x = 499
            if np.round(y) >= 500:
                y = 499
            x, y = np.round(x).astype(int), np.round(y).astype(int)
            if limit[0][0] <= x < limit[1][0] and limit[0][1] <= y < limit[1][1]:
                if 125 <= x < 250:
                    x -= 125
                elif 250 <= x < 375:
                    x -= 250
                elif 375 <= x < 500:
                    x -= 375
                if 125 <= y < 250:
                    y -= 125
                elif 250 <= y < 375:
                    y -= 250
                elif 375 <= y < 500:
                    y -= 375
                img = cv2.circle(img, center=(y, x), radius=5, color=1, thickness=1)
    return img


def test_det():
    det_model = cfcn(1)
    det_model.load_state_dict(torch.load('weights/weights_det.pth', map_location='cuda'))
    det_dataset = DetDataset("test")
    data_loaders = DataLoader(det_dataset, batch_size=1)
    det_model.eval()
    obj = {}
    with torch.no_grad():
        for x, label, name in tqdm(data_loaders):
            y = det_model(x).sigmoid()
            label = torch.squeeze(y).numpy()
            label = label.copy()
            id = int(name[0][3:-4])
            obj[id] = label
        return obj


def draw_img(_img, mat, box, limit, color):
    img = _img.copy()
    for i in range(mat.shape[0]):
        y = mat[i][0]
        x = mat[i][1]
        if limit is None:
            img = cv2.circle(img, center=(y, x), radius=5, color=1, thickness=1)
        else:
            if np.round(x) >= 500:
                x = 499
            if np.round(y) >= 500:
                y = 499
            x, y = np.round(x).astype(int), np.round(y).astype(int)
            if limit[0][0] <= x < limit[1][0] and limit[0][1] <= y < limit[1][1]:
                if 125 <= x < 250:
                    x -= 125
                elif 250 <= x < 375:
                    x -= 250
                elif 375 <= x < 500:
                    x -= 375
                if 125 <= y < 250:
                    y -= 125
                elif 250 <= y < 375:
                    y -= 250
                elif 375 <= y < 500:
                    y -= 375
                img = cv2.circle(img, center=(y, x), radius=7, color=color, thickness=1)
    for i in range(box.shape[0]):
        x = box[i, 0]
        y = box[i, 1]
        img = cv2.circle(img, center=(y, x), radius=3, color=color, thickness=-1)
    return img


def test(obj=None):
    model = cfcn(5)
    model.load_state_dict(torch.load('weights/weights_best.pth', map_location='cuda'))
    det_dataset = ClsDataset("test")
    data_loaders = DataLoader(det_dataset, batch_size=1)
    model.eval()
    if not os.path.exists("output/"):
        os.makedirs("output/epi/")
        os.makedirs("output/fib/")
        os.makedirs("output/inf/")
        os.makedirs("output/oth/")
    for files in os.listdir("output/epi"):
        os.remove("output/epi/" + files)
    for files in os.listdir("output/fib"):
        os.remove("output/fib/" + files)
    for files in os.listdir("output/inf"):
        os.remove("output/inf/" + files)
    for files in os.listdir("output/oth"):
        os.remove("output/oth/" + files)
    np.set_printoptions(threshold=np.inf)
    raw_path = 'CRCHistoPhenotypes_2016_04_28/Classification/'
    limit = np.array([
        [[0, 0], [125, 125]], [[0, 125], [125, 250]], [[0, 250], [125, 375]], [[0, 375], [125, 500]],
        [[125, 0], [250, 125]], [[125, 125], [250, 250]], [[125, 250], [250, 375]], [[125, 375], [250, 500]],
        [[250, 0], [375, 125]], [[250, 125], [375, 250]], [[250, 250], [375, 375]], [[250, 375], [375, 500]],
        [[375, 0], [500, 125]], [[375, 125], [500, 250]], [[375, 250], [500, 375]], [[375, 375], [500, 500]]
    ], dtype=np.int
    )

    orgs = []
    pred_boxs = []
    limits = []
    with torch.no_grad():
        for x, label, name in tqdm(data_loaders):
            y = model(x)
            y = F.softmax(y, dim=1)

            id = int(name[0][3:-4])
            if id % 16 == 0:
                n = id // 16
                m = 15
            else:
                n = id // 16 + 1
                m = id % 16 - 1

            img = np.array(Image.open('data/test/img/img{}.bmp'.format(id)))
            y_epi = torch.squeeze(y).numpy()[1]
            y_epi = y_epi.copy()
            if obj is not None:
                y_epi *= obj[id]
            pred_box, img_epi = non_max_suppression(y_epi, prob_thresh=0.45, length=125)
            pred_boxs.append(pred_box)
            epi_mat = loadmat(raw_path + "img{}/img{}_epithelial.mat".format(n, n))['detection']
            img_epi = draw_gt(epi_mat, img_epi, limit[m])
            orgs.append(epi_mat)
            limits.append(limit[m])
            plt.imsave('output/epi/{}'.format(name[0]), img_epi)
            # img_epi1 = draw_gt(epi_mat, y_epi, limit[m])
            plt.imsave('output/epi/y_{}'.format(name[0]), y_epi)
            img = draw_img(img, epi_mat, pred_box, limit[m], (255, 0, 0))

            y_fib = torch.squeeze(y).numpy()[2]
            y_fib = y_fib.copy()
            if obj is not None:
                y_fib *= obj[id]
            pred_box, img_fib = non_max_suppression(y_fib, prob_thresh=0.45, length=125)
            pred_boxs.append(pred_box)
            fib_mat = loadmat(raw_path + "img{}/img{}_fibroblast.mat".format(n, n))['detection']
            img_fib = draw_gt(fib_mat, img_fib, limit[m])
            orgs.append(fib_mat)
            limits.append(limit[m])
            plt.imsave('output/fib/{}'.format(name[0]), img_fib)
            img_fib1 = draw_gt(fib_mat, y_fib, limit[m])
            plt.imsave('output/fib/y_{}'.format(name[0]), img_fib1)
            img = draw_img(img, fib_mat, pred_box, limit[m], (0, 255, 0))

            y_inf = torch.squeeze(y).numpy()[3]
            y_inf = y_inf.copy()
            if obj is not None:
                y_inf *= obj[id]
            pred_box, img_inf = non_max_suppression(y_inf, prob_thresh=0.45, length=125)
            pred_boxs.append(pred_box)
            inf_mat = loadmat(raw_path + "img{}/img{}_inflammatory.mat".format(n, n))['detection']
            img_inf = draw_gt(inf_mat, img_inf, limit[m])
            orgs.append(inf_mat)
            limits.append(limit[m])
            plt.imsave('output/inf/{}'.format(name[0]), img_inf)
            img_inf1 = draw_gt(inf_mat, y_inf, limit[m])
            plt.imsave('output/inf/y_{}'.format(name[0]), img_inf1)
            img = draw_img(img, inf_mat, pred_box, limit[m], (0, 0, 255))

            y_oth = torch.squeeze(y).numpy()[4]
            y_oth = y_oth.copy()
            if obj is not None:
                y_oth *= obj[id]
            pred_box, img_oth = non_max_suppression(y_oth, prob_thresh=0.45, length=125)
            pred_boxs.append(pred_box)
            oth_mat = loadmat(raw_path + "img{}/img{}_others.mat".format(n, n))['detection']
            img_oth = draw_gt(oth_mat, img_oth, limit[m])
            orgs.append(oth_mat)
            limits.append(limit[m])
            plt.imsave('output/oth/{}'.format(name[0]), img_oth)
            img_oth1 = draw_gt(oth_mat, y_oth, limit[m])
            plt.imsave('output/oth/y_{}'.format(name[0]), img_oth1)
            img = draw_img(img, oth_mat, pred_box, limit[m], (0, 0, 0))
            plt.imsave('output/img/{}'.format(name[0]), img)

        precision, recall, f1, tp, pred_sum, gt_sum = get_metrics(orgs, pred_boxs, limits, size=125)
        print("pre:{}, rec:{}, f1:{}".format(precision, recall, f1))
        print("tp:{}, pred:{}, gt:{}".format(tp, pred_sum, gt_sum))
        print("*****")


if __name__ == '__main__':
    obj = test_det()
    test(obj)
