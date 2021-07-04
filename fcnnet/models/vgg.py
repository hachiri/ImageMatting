import os.path as osp
import fcn
import torch
import torchvision


def VGG16(pretrained=False):
    model = torchvision.models.vgg16(pretrained=False)
    if not pretrained:
        return model
    model_file_path = './models/vgg16_from_caffe.pth'
    if not osp.exists(model_file_path):
        model_file_path = _get_vgg16_pretrained_model()
    state_dict = torch.load(model_file_path)
    model.load_state_dict(state_dict)
    return model

def _get_vgg16_pretrained_model():
    return fcn.data.cached_download(
        url='https://download.pytorch.org/models/vgg16-397923af.pth',
        path=osp.expanduser('./models/vgg16_from_caffe.pth'),
        md5='aa75b158f4181e7f6230029eb96c1b13',
    )
