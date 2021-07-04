#!/usr/bin/env python

import argparse
import os, sys
import os.path as osp
import fcn
import numpy as np
import skimage.io
import torch
from torch.autograd import Variable
here = osp.dirname(osp.abspath(__file__))
sys.path.append(here)
import fcnnet
import tqdm

def main():
    n_class = 2
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--img', type=str, required=True, help='image data directory')
    parser.add_argument('--mask', type=str, required=True, help='mask data directory')
    parser.add_argument('-m', '--model', type=str, default='models/FCN8s.pth.tar', help='model path')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    root = osp.expanduser('./dataset')
    val_loader = torch.utils.data.DataLoader(
        fcnnet.datasets.ImageMatting(root, img_dir_path=args.img, mask_dir_path=args.mask,
                                     split='val', transform=True),
        batch_size=1, shuffle=True)

    model = fcnnet.models.FCN8sAtOnce(n_class=n_class)
    model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, args.model))
    model_data = torch.load(args.model, map_location=torch.device('cpu'))
    try:
        model.load_state_dict(model_data)
    except Exception:
        model.load_state_dict(model_data['model_state_dict'])
    model.eval()

    print('==> Evaluating with TestImage')
    visualizations = []
    label_trues, label_preds = [], []

    for batch_idx, (data, target) in tqdm.tqdm(enumerate(val_loader), total=len(val_loader),
                                               ncols=80, leave=False):
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            score = model(data)

            imgs = data.data.cpu()
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu()
            for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                img, lt = val_loader.dataset.untransform(img, lt)
                label_trues.append(lt)
                label_preds.append(lp)
                if len(visualizations) < 9:
                    viz = fcn.utils.visualize_segmentation(
                        lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class)
                    visualizations.append(viz)
    metrics = fcnnet.utils.label_accuracy_score(
        label_trues, label_preds, n_class=n_class)
    metrics = np.array(metrics)
    metrics *= 100
    print('''\
    Accuracy: {0}
    Accuracy Class: {1}
    Mean IU: {2}
    FWAV Accuracy: {3}'''.format(*metrics))

    viz = fcn.utils.get_tile_image(visualizations)
    skimage.io.imsave('./models/viz_evaluate.png', viz)

if __name__ == '__main__':
    main()
