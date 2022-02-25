import natsort, glob, pickle, torch
from collections import OrderedDict
import numpy as np
import os
import sys,io
import options.options as option
from models import create_model
from imresize import imresize

import Measure

def find_files(wildcard): return natsort.natsorted(glob.glob(wildcard, recursive=True))

from PIL import Image

from test import load_model, fiFindByWildcard, imread

def pickleRead(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# Convert to tensor

def t(array): return torch.Tensor(np.expand_dims(array.transpose([2, 0, 1]), axis=0).astype(np.float32)) / 255

# convert to image
def rgb(t): return (np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(np.uint8)

def load_srflow_model():
    conf_path = './confs/SRFlow_CelebA_8X.yml'
    model, opt = load_model(conf_path)
    return model


def get_test_path(kind):
    if kind =='lr':
        return './data/test/lr'
    return './data/test/gt'