#!/usr/bin/env python

import warnings
warnings.filterwarnings('ignore')
import argparse
import os, sys
import numpy as np
import os.path as osp
import skimage.io
import torch
from torch.autograd import Variable
here = osp.dirname(osp.abspath(__file__))
sys.path.append(here)
import fcnnet
import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('-f', '--file', type=str, required=True, help='image path')
    parser.add_argument('-m', '--model', type=str, default='models/FCN8s.pth.tar', help='model path')
    parser.add_argument('-c', '--change', type=str, help='B=Blue, W=White')
    parser.add_argument('-bg', '--background', type=str, help='Background image path')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    data = fcnnet.utils.get_standard_pic(args.file)
    data = torch.unsqueeze(data, 0)
    # data = data.cuda()

    n_class = 2
    model = fcnnet.models.FCN8sAtOnce(n_class=n_class)
    # model = model.cuda()
    # model_data = torch.load(args.model)
    model_data = torch.load(model, map_location='cpu')
    try:
        model.load_state_dict(model_data)
    except Exception:
        model.load_state_dict(model_data['model_state_dict'])
    model.eval()

    with torch.no_grad():
        data = Variable(data)
        score = model(data)

        imgs = data.data.cpu()
        lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
        for img, lp in zip(imgs, lbl_pred):
            img = fcnnet.utils.untransform(img)
            img = torch.from_numpy(img.copy())
            mask = lp.astype(np.uint8)
            lp = torch.from_numpy(lp)

            if args.change == 'B':
                img_changed = fcnnet.utils.get_blue(img, lp)
            elif args.change == 'W':
                img_changed = fcnnet.utils.get_white(img, lp)
            elif args.background:
                img_changed = fcnnet.utils.change_bg(img, lp, pic_path=args.background)
            else:
                img_changed = fcnnet.utils.change_bg(img, lp)

    file_name = osp.splitext(osp.split(args.file)[1])[0]
    skimage.io.imsave("./tmp/" + file_name + "_changed.jpg", img_changed)
    skimage.io.imsave("./tmp/" + file_name + "_mask.jpg", mask*255)
    # skimage.io.imsave(pic_path.rsplit('.', maxsplit=1)[0] + '_white.png', change1)
    # skimage.io.imsave(pic_path.rsplit('.', maxsplit=1)[0] + '_changed_bg.png', change2)
    print('Change Finished')

if __name__ == '__main__':
    main()
