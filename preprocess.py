from scipy.io import loadmat
import cv2
import numpy as np
import os
import PIL.Image as Image

radius = 2  # r = 3


def to_label(mat, limit, label=None, color=1, size=125):
    if label is None:
        label = np.zeros((size, size), dtype=np.uint8)
    length = mat.shape[0]
    for i in range(length):
        y = mat[i][0]
        x = mat[i][1]

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
            cv2.circle(label, center=(y, x), radius=2, color=color, thickness=-1)
    return label


def detection():
    raw_path = 'CRCHistoPhenotypes_2016_04_28/Detection/'
    new_path = 'data/'

    for num in range(100):
        image_name = "img{}/img{}".format(num + 1, num + 1)
        image = cv2.imread(raw_path + image_name + ".bmp")
        mat = loadmat(raw_path + image_name + "_detection.mat")
        mat_det = mat['detection']

        limit = np.array([
            [[0, 0], [125, 125]], [[0, 125], [125, 250]], [[0, 250], [125, 375]], [[0, 375], [125, 500]],
            [[125, 0], [250, 125]], [[125, 125], [250, 250]], [[125, 250], [250, 375]], [[125, 375], [250, 500]],
            [[250, 0], [375, 125]], [[250, 125], [375, 250]], [[250, 250], [375, 375]], [[250, 375], [375, 500]],
            [[375, 0], [500, 125]], [[375, 125], [500, 250]], [[375, 250], [500, 375]], [[375, 375], [500, 500]]
        ], dtype=np.int
        )
        for i in range(16):
            id = num * 16 + i + 1
            img = image[limit[i][0][0]:limit[i][1][0], limit[i][0][1]:limit[i][1][1]]
            cv2.imwrite(new_path + 'det_img/img{}.bmp'.format(id), img)
            label = np.zeros((125, 125), dtype=np.uint8).copy()
            label = to_label(mat_det, limit[i], label=label, color=225)
            cv2.imwrite(new_path + 'det_label/img{}.bmp'.format(id), label)


def classification():
    raw_path = 'CRCHistoPhenotypes_2016_04_28/Classification/'
    new_path = 'data/'

    for num in range(100):
        image_name = "img{}/img{}".format(num + 1, num + 1)
        image = cv2.imread(raw_path + image_name + ".bmp")
        epi_mat = loadmat(raw_path + image_name + "_epithelial.mat")['detection']
        fib_mat = loadmat(raw_path + image_name + "_fibroblast.mat")['detection']
        inf_mat = loadmat(raw_path + image_name + "_inflammatory.mat")['detection']
        oth_mat = loadmat(raw_path + image_name + "_others.mat")['detection']
        limit = np.array([
            [[0, 0], [125, 125]], [[0, 125], [125, 250]], [[0, 250], [125, 375]], [[0, 375], [125, 500]],
            [[125, 0], [250, 125]], [[125, 125], [250, 250]], [[125, 250], [250, 375]], [[125, 375], [250, 500]],
            [[250, 0], [375, 125]], [[250, 125], [375, 250]], [[250, 250], [375, 375]], [[250, 375], [375, 500]],
            [[375, 0], [500, 125]], [[375, 125], [500, 250]], [[375, 250], [500, 375]], [[375, 375], [500, 500]]
        ], dtype=np.int
        )
        for i in range(16):
            id = num * 16 + i + 1
            img = image[limit[i][0][0]:limit[i][1][0], limit[i][0][1]:limit[i][1][1]]
            cv2.imwrite(new_path + 'cls_img/img{}.bmp'.format(id), img)
            label = np.zeros((125, 125), dtype=np.uint8).copy()
            label = to_label(epi_mat, limit[i], label=label, color=255)
            label = to_label(fib_mat, limit[i], label=label, color=255)
            label = to_label(inf_mat, limit[i], label=label, color=255)
            label = to_label(oth_mat, limit[i], label=label, color=255)
            cv2.imwrite(new_path + 'cls_label/img{}.bmp'.format(id), label)


def fun():
    raw_path = 'CRCHistoPhenotypes_2016_04_28/Classification/'
    new_path = 'data/'
    kind_path = ['train/', 'val/', 'test/']
    il_path = ['img/', 'epi_label/', 'fib_label/', 'inf_label/', 'oth_label/', 'label/']
    if not os.path.exists(new_path + 'img/'):
        for k in kind_path:
            for il in il_path:
                os.makedirs(new_path + k + il)
                if k == 'train/':
                    os.makedirs(new_path + k + il + 'output/')

    index = np.linspace(1, 1600, 1600)
    np.random.seed(2)
    with open("bad.txt", "r") as file:
        bad = eval(file.readline())
    index = np.setdiff1d(index, bad)
    test = np.random.choice(index, 109, replace=False)
    index = np.setdiff1d(index, test)
    val = np.random.choice(index, 109, replace=False)
    train = np.setdiff1d(index, val)

    for num in range(100):
        image_name = "img{}/img{}".format(num + 1, num + 1)
        image = cv2.imread(raw_path + image_name + ".bmp")
        epi_mat = loadmat(raw_path + image_name + "_epithelial.mat")['detection']
        fib_mat = loadmat(raw_path + image_name + "_fibroblast.mat")['detection']
        inf_mat = loadmat(raw_path + image_name + "_inflammatory.mat")['detection']
        oth_mat = loadmat(raw_path + image_name + "_others.mat")['detection']
        limit = np.array([
            [[0, 0], [125, 125]], [[0, 125], [125, 250]], [[0, 250], [125, 375]], [[0, 375], [125, 500]],
            [[125, 0], [250, 125]], [[125, 125], [250, 250]], [[125, 250], [250, 375]], [[125, 375], [250, 500]],
            [[250, 0], [375, 125]], [[250, 125], [375, 250]], [[250, 250], [375, 375]], [[250, 375], [375, 500]],
            [[375, 0], [500, 125]], [[375, 125], [500, 250]], [[375, 250], [500, 375]], [[375, 375], [500, 500]]
        ], dtype=np.int
        )
        for i in range(16):
            id = num * 16 + i + 1
            if id not in bad:
                if id in test:
                    kind = 'test'
                elif id in train:
                    kind = 'train'
                else:
                    kind = 'val'
                img = image[limit[i][0][0]:limit[i][1][0], limit[i][0][1]:limit[i][1][1]]
                epi_label = to_label(epi_mat, limit[i])
                fib_label = to_label(fib_mat, limit[i])
                inf_label = to_label(inf_mat, limit[i])
                oth_label = to_label(oth_mat, limit[i])
                cv2.imwrite('{}/{}/epi_label/img{}.bmp'.format(new_path, kind, id), epi_label)
                cv2.imwrite('{}/{}/fib_label/img{}.bmp'.format(new_path, kind, id), fib_label)
                cv2.imwrite('{}/{}/inf_label/img{}.bmp'.format(new_path, kind, id), inf_label)
                cv2.imwrite('{}/{}/oth_label/img{}.bmp'.format(new_path, kind, id), oth_label)
                cv2.imwrite('{}/{}/img/img{}.bmp'.format(new_path, kind, id), img)

    imgs = []
    masks = []
    for img_name in os.listdir("data/train/img"):
        if img_name == 'output':
            continue
        img = np.array(Image.open(os.path.join("data/train/img", img_name)))
        img_epi = np.array(Image.open(os.path.join("data/train/epi_label", img_name)))
        img_fib = np.array(Image.open(os.path.join("data/train/fib_label", img_name)))
        img_inf = np.array(Image.open(os.path.join("data/train/inf_label", img_name)))
        img_oth = np.array(Image.open(os.path.join("data/train/oth_label", img_name)))
        img_mask = [img_epi, img_fib, img_inf, img_oth]
        imgs.append(img)
        masks.append(img_mask)

    for i in range(len(imgs)):
        img = imgs[i]
        img_mask = masks[i]
        img_epi = img_mask[0]
        img_fib = img_mask[1]
        img_inf = img_mask[2]
        img_oth = img_mask[3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite('data/train/img/output/img{}.bmp'.format(i + 1), img)
        cv2.imwrite('data/train/epi_label/output/img{}.bmp'.format(i + 1), img_epi)
        cv2.imwrite('data/train/fib_label/output/img{}.bmp'.format(i + 1), img_fib)
        cv2.imwrite('data/train/inf_label/output/img{}.bmp'.format(i + 1), img_inf)
        cv2.imwrite('data/train/oth_label/output/img{}.bmp'.format(i + 1), img_oth)


def get_bad():
    bad = []
    for labels in os.listdir('data/det_label'):
        det_label = np.array(Image.open('data/det_label/' + labels)).sum()
        cla_label = np.array(Image.open('data/cls_label/' + labels)).sum()
        if cla_label < 0.8 * det_label or cla_label == 0:
            bad.append(int(labels[3:-4]))
    bad = sorted(bad)
    with open("bad.txt", "w") as file:
        file.write(str(bad))
    print(bad)
    all = 1600 - len(bad)
    val = int(all * 0.1)
    test = val
    train = all - val - test
    print("all:{}, train:{} , test:{}, val:{}".format(all, train, test, val))
    with open("bad.txt", "r") as file:
        bad = eval(file.readline())
    print(bad)


def get_pretrained():
    raw_path = 'CRCHistoPhenotypes_2016_04_28/Detection/'
    new_path = 'data/detection/'
    kind_path = ['train/', 'val/', 'test/']
    il_path = ['img/', 'label/']
    if not os.path.exists(new_path):
        for k in kind_path:
            for il in il_path:
                os.makedirs(new_path + k + il)

    index = np.linspace(1, 1600, 1600)
    np.random.seed(2)
    with open("bad.txt", "r") as file:
        bad = eval(file.readline())
    index = np.setdiff1d(index, bad)
    test = np.random.choice(index, 109, replace=False)
    index = np.setdiff1d(index, test)
    val = np.random.choice(index, 109, replace=False)
    index = np.linspace(1, 1600, 1600)
    train = np.setdiff1d(index, val)
    train = np.setdiff1d(train, test)

    for num in range(100):
        image_name = "img{}/img{}".format(num + 1, num + 1)
        image = cv2.imread(raw_path + image_name + ".bmp")
        mat = loadmat(raw_path + image_name + "_detection.mat")
        mat_det = mat['detection']

        limit = np.array([
            [[0, 0], [125, 125]], [[0, 125], [125, 250]], [[0, 250], [125, 375]], [[0, 375], [125, 500]],
            [[125, 0], [250, 125]], [[125, 125], [250, 250]], [[125, 250], [250, 375]], [[125, 375], [250, 500]],
            [[250, 0], [375, 125]], [[250, 125], [375, 250]], [[250, 250], [375, 375]], [[250, 375], [375, 500]],
            [[375, 0], [500, 125]], [[375, 125], [500, 250]], [[375, 250], [500, 375]], [[375, 375], [500, 500]]
        ], dtype=np.int
        )
        for i in range(16):
            id = num * 16 + i + 1
            # if id not in bad:
            if id in test:
                kind = 'test'
            elif id in train:
                kind = 'train'
            else:
                kind = 'val'
            img = image[limit[i][0][0]:limit[i][1][0], limit[i][0][1]:limit[i][1][1]]
            cv2.imwrite(new_path + '{}/img/img{}.bmp'.format(kind, id), img)
            label = np.zeros((125, 125), dtype=np.uint8).copy()
            label = to_label(mat_det, limit[i], label=label, color=1)
            cv2.imwrite(new_path + '{}/label/img{}.bmp'.format(kind, id), label)
            if kind != 'train':
                cv2.imwrite(new_path + '{}/label/img{}.bmp'.format('train', id), label)
                cv2.imwrite(new_path + '{}/img/img{}.bmp'.format('train', id), img)


if __name__ == '__main__':
    detection()
    classification()
    get_bad()
    fun()
    get_pretrained()
