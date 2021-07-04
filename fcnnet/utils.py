import numpy as np
import PIL.Image
import torch
from fcnnet.datasets.voc import ImageMatting

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def label_accuracy_score(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc

def transform(img):
    img = img[:, :, ::-1].copy()  # RGB -> BGR
    img -= ImageMatting.mean_bgr
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img

def untransform(img):
    img = img.numpy()
    img = img.transpose(1, 2, 0)
    img += ImageMatting.mean_bgr
    img = img.astype(np.uint8)
    img = img[:, :, ::-1].copy()
    return img

def get_standard_pic(pic_path):
    img = PIL.Image.open(pic_path)
    size = 600, 800
    img.thumbnail(size)
    # file_name = osp.splitext(osp.split(pic_path)[1])[0]
    # img.save("./tmp/" + file_name + "_processed.jpg")
    img = np.array(img, dtype=np.float64)
    img_tensor = transform(img)
    return img_tensor

def get_white(img, lbl):
    img[lbl == 0] = 255
    return img

def get_blue(img, lbl):
    img[:,:,0][lbl == 0] = 135
    img[:,:,1][lbl == 0] = 206
    img[:,:,2][lbl == 0] = 235
    return img

def get_red(img, lbl):
    img[:,:,0][lbl == 0] = 227
    img[:,:,1][lbl == 0] = 23
    img[:,:,2][lbl == 0] = 13
    return img

def get_rgb(img, lbl, rgb):
    img[:,:,0][lbl == 0] = rgb[0]
    img[:,:,1][lbl == 0] = rgb[1]
    img[:,:,2][lbl == 0] = rgb[2]
    return img

def change_bg(img, lbl, pic_path='bg/top1.jpg'):
    a = lbl == 1
    a = a.expand(3, -1, -1).permute(1, 2, 0)
    mask1 = img * a

    back_img = PIL.Image.open(pic_path)
    back_img = np.array(back_img, dtype=np.uint8)
    back_img = torch.from_numpy(back_img)

    a = lbl == 0
    a = a.expand(3, -1, -1).permute(1, 2, 0)
    mask2 = back_img * a

    return mask1 + mask2