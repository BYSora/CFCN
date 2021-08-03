import numpy as np
import cv2
import PIL.Image as Image
import os
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

epsilon = 1e-7

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.814, 0.676, 0.842], [0.143, 0.176, 0.104])
])

y_train_transforms = transforms.ToTensor()
y_transforms = transforms.Compose([
    transforms.ToTensor()
])


def make_dataset(root):
    img_pair = []
    if root == 'test' or root == 'val':
        for img_name in os.listdir("data/detection/" + root + "/img"):
            img = os.path.join("data/detection/" + root + "/img", img_name)
            label = os.path.join("data/detection/" + root + "/label", img_name)
            img_pair.append((img, label, img_name))
        return img_pair
    else:
        for img_name in os.listdir("data/detection/" + root + "/img"):
            if 'output' in img_name:
                continue
            img = os.path.join("data/detection/train/img", img_name)
            label = os.path.join("data/detection/train/label", img_name)
            img_pair.append((img, label, img_name))
        return img_pair


def make_cls_dataset(root):

    img_pair = []
    if root == 'test' or root == 'val':
        for img_name in os.listdir("data/" + root + "/img"):
            img = os.path.join("data/" + root + "/img", img_name)
            epi_label = os.path.join("data/" + root + "/epi_label", img_name)
            fib_label = os.path.join("data/" + root + "/fib_label", img_name)
            inf_label = os.path.join("data/" + root + "/inf_label", img_name)
            oth_label = os.path.join("data/" + root + "/oth_label", img_name)
            img_pair.append((img, epi_label, fib_label, inf_label, oth_label, img_name))
        return img_pair
    else:
        for img_name in os.listdir("data/" + root + "/img/output"):
            if 'output' in img_name:
                continue
            img = os.path.join("data/train/img/output", img_name)
            epi_label = os.path.join("data/train/epi_label/output", img_name)
            fib_label = os.path.join("data/train/fib_label/output", img_name)
            inf_label = os.path.join("data/train/inf_label/output", img_name)
            oth_label = os.path.join("data/train/oth_label/output", img_name)
            img_pair.append((img, epi_label, fib_label, inf_label, oth_label, img_name))
        return img_pair


class DetDataset(Dataset):
    def __init__(self, root, augmentation=None):
        img_pair = make_dataset(root)
        self.img_pair = img_pair
        self.augmentation = augmentation
        self.x_transforms = x_transforms
        self.y_transforms = y_transforms
        self.y_train_transforms = y_train_transforms
        self.root = root

    def __getitem__(self, item):
        x_path, y_path, name = self.img_pair[item]
        img_x = Image.open(x_path)
        img_x = np.array(img_x)
        img_y = Image.open(y_path)
        img_y = np.array(img_y)

        if self.augmentation:
            sample = self.augmentation(image=img_x, mask=img_y)
            img_x, img_y = sample['image'], sample['mask']
        if self.x_transforms is not None:
            img_x = self.x_transforms(img_x)
        if self.y_transforms is not None:
            img_y = torch.as_tensor(img_y, dtype=torch.float)
            img_y = torch.unsqueeze(img_y, 0)
        return img_x, img_y, name

    def __len__(self):
        return len(self.img_pair)


class ClsDataset(Dataset):
    def __init__(self, root, augmentation=None):
        img_pair = make_cls_dataset(root)
        self.img_pair = img_pair
        self.augmentation = augmentation
        self.x_transforms = x_transforms
        self.y_transforms = y_transforms
        self.y_train_transforms = y_train_transforms
        self.root = root

    def __getitem__(self, item):
        x_path, epi_path, fib_path, inf_path, oth_path, name = self.img_pair[item]
        img_x = Image.open(x_path)
        img_y = 0
        img_epi = Image.open(epi_path)
        img_fib = Image.open(fib_path)
        img_inf = Image.open(inf_path)
        img_oth = Image.open(oth_path)
        img_x = np.array(img_x)
        img_epi = np.array(img_epi)
        img_fib = np.array(img_fib)
        img_inf = np.array(img_inf)
        img_oth = np.array(img_oth)
        if self.augmentation:
            img_mask = [img_epi, img_fib, img_inf, img_oth]
            sample = self.augmentation(image=img_x, masks=img_mask)
            img_x, img_mask = sample['image'], sample['masks']
            img_epi = img_mask[0]
            img_fib = img_mask[1]
            img_inf = img_mask[2]
            img_oth = img_mask[3]
        if self.x_transforms is not None:
            img_x = self.x_transforms(img_x)
        if self.y_transforms is not None:

            img_bg = img_epi + img_fib + img_inf + img_oth
            img_epi[img_bg > 1] = 0
            img_fib[img_bg > 1] = 0
            img_inf[img_bg > 1] = 0
            img_oth[img_bg > 1] = 0
            img_bg[img_bg > 1] = 1

            img_epi = torch.as_tensor(img_epi, dtype=torch.float)
            img_epi = torch.unsqueeze(img_epi, 0)
            img_fib = torch.as_tensor(img_fib, dtype=torch.float)
            img_fib = torch.unsqueeze(img_fib, 0)
            img_inf = torch.as_tensor(img_inf, dtype=torch.float)
            img_inf = torch.unsqueeze(img_inf, 0)
            img_oth = torch.as_tensor(img_oth, dtype=torch.float)
            img_oth = torch.unsqueeze(img_oth, 0)
            img_bg = 1 - img_bg
            img_bg = torch.as_tensor(img_bg, dtype=torch.float)
            img_bg = torch.unsqueeze(img_bg, 0)
            img_y = torch.cat([img_bg, img_epi, img_fib, img_inf, img_oth])
        return img_x, img_y, name

    def __len__(self):
        return len(self.img_pair)


def non_max_suppression(img, prob_thresh, r=5, overlap_thresh=0.1, length=250):
    x1s = []
    y1s = []
    x2s = []
    y2s = []
    scores = []
    xs = []
    ys = []
    img[img < prob_thresh] = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] > 0:
                x1 = max(i - r, 0)
                y1 = max(j - r, 0)
                x2 = min(i + r, img.shape[0] - 1)
                y2 = min(j + r, img.shape[1] - 1)
                s = 0.0
                for x in range(i - 2, i + 3):
                    for y in range(j - 2, j + 3):
                        if 0 <= x < 125 and 0 <= y < 125 and img[x][y] > 0:
                            s += 1
                if s >= 10:
                    xs.append(i)
                    ys.append(j)
                    x1s.append(x1)
                    y1s.append(y1)
                    x2s.append(x2)
                    y2s.append(y2)
                    scores.append(img[i][j])
    x1s = np.array(x1s)
    y1s = np.array(y1s)
    x2s = np.array(x2s)
    y2s = np.array(y2s)
    xs = np.array(xs)
    ys = np.array(ys)
    box = np.concatenate((xs.reshape((xs.shape[0], 1)), ys.reshape((ys.shape[0], 1))), axis=1)
    scores = np.array(scores)
    areas = (x2s - x1s + 1) * (y2s - y1s + 1)
    keep = []
    order = scores.argsort()[::-1]
    pred_map = np.zeros((length, length), dtype=np.uint8).copy()
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1s[i], x1s[order[1:]])
        yy1 = np.maximum(y1s[i], y1s[order[1:]])
        xx2 = np.minimum(x2s[i], x2s[order[1:]])
        yy2 = np.minimum(y2s[i], y2s[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= overlap_thresh)[0]
        order = order[inds + 1]

    for k in keep:
        cv2.circle(pred_map, center=(ys[k], xs[k]), radius=2, color=1, thickness=-1)
    return box[keep], pred_map


def get_metrics(gts, preds, limits, r=5, size=250):
    gt_sum = 0
    pred_sum = 0
    tp = 0
    for item in range(len(gts)):
        gt = gts[item]
        gt_map = np.zeros((size, size), dtype=np.uint8).copy()
        for i in range(gt.shape[0]):
            y = gt[i][0]
            x = gt[i][1]

            if limits is None:
                cv2.circle(gt_map, center=(y, x), radius=r, color=1, thickness=-1)
                gt_sum += 1
            else:
                if np.round(x) >= 500:
                    x = 499
                if np.round(y) >= 500:
                    y = 499

                x, y = np.round(x).astype(int), np.round(y).astype(int)
                if limits[item][0][0] <= x < limits[item][1][0] and limits[item][0][1] <= y < limits[item][1][1]:
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
                    cv2.circle(gt_map, center=(y, x), radius=r, color=1, thickness=-1)
                    gt_sum += 1

        pred = preds[item]
        pred_map = np.zeros((size, size), dtype=np.uint8)
        for i in range(pred.shape[0]):
            x = pred[i, 0]
            y = pred[i, 1]
            pred_map[x, y] = 1
            pred_sum += 1

        result_map = gt_map * pred_map
        tp += result_map.sum()

    precision = tp / (pred_sum + epsilon)
    recall = tp / (gt_sum + epsilon)
    f1_score = 2 * (precision * recall / (precision + recall + epsilon))

    return precision, recall, f1_score, tp, pred_sum, gt_sum


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=8, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = F.softmax(_input, dim=1)
        pt = pt.clamp(min=0.000001, max=0.999999)
        pt = pt[:, 1:5, :, :]
        target = target[:, 1:5, :, :]
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) \
               - pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        loss = torch.mean(loss)

        return loss
