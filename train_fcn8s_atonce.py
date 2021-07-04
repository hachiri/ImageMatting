#!/usr/bin/env python

import argparse
import datetime
import os, sys
import os.path as osp
here = osp.dirname(osp.abspath(__file__))
sys.path.append(here)
import torch
import torch.nn as nn
import yaml
import fcnnet

def get_parameters(model, bias=False):
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.Sequential,
        fcnnet.models.FCN8sAtOnce,
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None
        elif isinstance(m, modules_skipped):
            continue
        else:
            raise ValueError('Unexpected module: %s' % str(m))

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-g', '--gpu', type=int, required=True, help='GPU ID')
    parser.add_argument('--img', type=str, required=True, help='image data directory')
    parser.add_argument('--mask', type=str, required=True, help='mask data directory')
    parser.add_argument('--resume', help='checkpoint path')
    parser.add_argument(
        '--max-iteration', type=int, default=100000, help='max iteration'
    )
    parser.add_argument(
        '--lr', type=float, default=1.0e-10, help='learning rate',
    )
    parser.add_argument(
        '--weight-decay', type=float, default=0.0005, help='weight decay',
    )
    parser.add_argument(
        '--momentum', type=float, default=0.99, help='momentum',
    )
    args = parser.parse_args()

    # write log files to ./logs/[timestamp]
    now = datetime.datetime.now()
    args.out = osp.join(here, 'logs', now.strftime('%Y%m%d_%H%M%S.%f'))
    os.makedirs(args.out)
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    # set GPU and random_state
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    # create dataset
    root = osp.expanduser('./dataset')
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        fcnnet.datasets.ImageMatting(root, img_dir_path=args.img, mask_dir_path=args.mask,
                                     split='train', transform=True),
        batch_size=1, shuffle=False, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        fcnnet.datasets.ImageMatting(root, img_dir_path=args.img, mask_dir_path=args.mask,
                                     split='val', transform=True),
        batch_size=1, shuffle=True, **kwargs)

    # create model
    model = fcnnet.models.FCN8sAtOnce(n_class=2)
    start_epoch = 0
    start_iteration = 0
    if args.resume:
        # inherit model data from the checkpoint
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        # train FCN-8s model from zero, but VGG16 model is pretrained model
        vgg16 = fcnnet.models.VGG16(pretrained=True)
        model.copy_params_from_vgg16(vgg16)
    model = model.cuda()

    # set optimizer
    optim = torch.optim.SGD(
        [
            {'params': get_parameters(model, bias=False)},
            {'params': get_parameters(model, bias=True),
             'lr': args.lr * 2, 'weight_decay': 0},
        ],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    if args.resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    trainer = fcnnet.Trainer(
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        out=args.out,
        max_iter=args.max_iteration,
        interval_validate=4000,
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()

if __name__ == '__main__':
    main()
