import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil

from torch.autograd import Function
import numpy as np
import cv2


def save_checkpoint(state, is_best, save_folder, filename = 'checkpoint.pth.tar'):
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    torch.save(state, save_folder + '/' + filename)
    if is_best:
        shutil.copyfile(save_folder + '/' + filename,
		save_folder + '/' + 'model_best.pth.tar')