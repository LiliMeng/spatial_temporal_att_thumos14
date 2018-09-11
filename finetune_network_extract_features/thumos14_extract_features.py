'''
Extract features for thumos14 dataset

Author: Lili Meng
Date: August 28th, 2018
'''
import numpy as np
import pickle
import os
from PIL import Image
import time
from tqdm import tqdm
import shutil
from random import randint
import argparse
import os

import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

import dataloader.spatial_dataloader
from utils import *
from network import *
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='THUMOS14 spatial stream on resnet101')
parser.add_argument('--num_frames', default=50, type=int, metavar='N', help='number of classes in the dataset')


class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])

    def forward(self, x):
        x = self.features(x)
        
        return x


def load_frames(new_img_list, img_list, video_root_path, num_frames=15):

    lines = [line.strip() for line in open(img_list).readlines()]

    frames_for_all_video_list = []
    labels_for_all_video_list = []
    frames_names_for_all_video_list = []

    less_frame_count = 0
    annot_img_index_list = []
    entire_img_index_list = []

    for line in lines:

        video_name = line.split(' ')[0]
        label = int(line.split(' ')[1])
        start_frame = int(float(line.split(' ')[2])*10)
        end_frame = int(float(line.split(' ')[3])*10)
        num_frames_entire_video = int(line.split(' ')[4])
        total_num_imgs_per_video = end_frame-start_frame

        video_path = os.path.join(video_root_path, video_name)

        
        img_interval = int((total_num_imgs_per_video/num_frames))
        
        if img_interval != 0:
            img_index_list = list(range(start_frame, end_frame+1, img_interval))
        else:
            img_index_list = list(range(start_frame, end_frame+1))

        if total_num_imgs_per_video > num_frames:
            img_index_list = img_index_list[0:num_frames]
            
        if total_num_imgs_per_video <= num_frames:
            less_frame_count +=1 
        
            final_start_frame = int(start_frame - int(0.5*(50-total_num_imgs_per_video)))
            final_end_frame = int(end_frame+1 + int(0.5*(50-total_num_imgs_per_video)))
            
            if final_start_frame < 1:
                final_start_frame =1
             #   final_end_frame = int(end_frame+1+int(0.5*(50-total_num_imgs_per_video))) +(1-tmp)

            img_index_list_before = list(range(final_start_frame, start_frame))
            img_index_list_after = list(range(end_frame+1, final_end_frame))
            final_img_index_list = img_index_list_before + img_index_list + img_index_list_after
            
        else:
            final_start_frame = start_frame-10*img_interval
            final_end_frame = end_frame + 10*img_interval

            tmp = final_start_frame
            if final_start_frame < 1:
                final_start_frame =1
                #final_end_frame = end_frame + 10*img_interval +(1-tmp)

            img_index_list_before = list(range(final_start_frame, start_frame, img_interval))
            img_index_list_after = list(range(end_frame+1, final_end_frame+1, img_interval))

          
            if final_end_frame > num_frames_entire_video:

                final_end_frame = end_frame +10

                if final_end_frame > num_frames_entire_video:
                    img_index_list_after = list(range(end_frame+1, num_frames_entire_video))
                else:
                    img_index_list_after = list(range(end_frame+1, final_end_frame+10))

            final_img_index_list = img_index_list_before + img_index_list + img_index_list_after

        if len(final_img_index_list)< 50:
            appending_after_list = list(range(final_end_frame+1, final_end_frame+1+50-len(final_img_index_list)))
            for i in range(len(appending_after_list)):
                if appending_after_list[i] > num_frames_entire_video:
                    appending_after_list[i] = num_frames_entire_video

            final_img_index_list += appending_after_list
        else:
            final_img_index_list = final_img_index_list[0:50]


        print("len(final_img_index_list): ", len(final_img_index_list))
     
        assert(len(final_img_index_list)==50)
        
        imgs_per_video_list = []
        imgs_names_per_video_list = []
        label_per_video_list = []
        exist_frame_count = 0
        print("final_img_index_list: ", final_img_index_list)
        for i in range(0, len(final_img_index_list)):

            img_name = os.path.join(video_path,  str('%05d'%(final_img_index_list[i])) + '.jpg')

            if os.path.isfile(img_name):
                exist_frame_count +=1
            else:
                img_name = os.path.join(video_path, str('%05d'%(final_img_index_list[exist_frame_count-1])) + '.jpg')

            img = Image.open(img_name).convert('RGB')

            try:
                imgs_per_video_list.append(img)
                imgs_names_per_video_list.append(img_name) 
            except:
                print(os.path.join(path, str('%05d'%(index)) + '.jpg'))
                img.close()

        frames_for_all_video_list.append(imgs_per_video_list)
        labels_for_all_video_list.append(label)
        frames_names_for_all_video_list.append(imgs_names_per_video_list)

    return annot_img_index_list, entire_img_index_list, frames_for_all_video_list, frames_names_for_all_video_list, labels_for_all_video_list

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(logits, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = 1
    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()


    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
         correct_k = correct[:k].view(-1).float().sum(0)
         res.append(correct_k.mul_(100.0 / batch_size))
    return res

def main():

	global arg
	arg = parser.parse_args()
	print(arg)

	# 1. build model
	print ('==> Build model and setup loss and optimizer')
	# build model
	model = resnet101(pretrained=True, channel=3).cuda()

	#Loss function and optimizer
	criterion = nn.CrossEntropyLoss().cuda()
	optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9)
	#self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
	scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1,verbose=True)

	model_resume_path ='./record/spatial/valid_test_best/model_best.pth.tar'
	# 2. load pretrained model
	if os.path.isfile(model_resume_path):
		print("==> loading checkpoint '{}'".format(model_resume_path))
		checkpoint = torch.load(model_resume_path)
		start_epoch = checkpoint['epoch']
		best_prec1 = checkpoint['best_prec1']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})".format(model_resume_path, checkpoint['epoch'], best_prec1))
	else:
		print("==> no checkpoint found at '{}'".format(model_resume_path))

	# 3. prepare input data including load all imgs and preprocessing, prepare input tensor
	transform = transforms.Compose([
				transforms.Scale([224, 224]),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

	annot_img_index_list, entire_img_index_list, all_frames, all_frame_names, all_labels = load_frames(new_img_list = "new_file_with_start_end_frame_test.txt",
                                            img_list = "./thumos14_list/final_thumos_14_20_one_label_temporal_test_with_img_num.txt",
                                            video_root_path = "/media/dataDisk/THUMOS14/THUMOS14_video/thumos14_preprocess/test/frames_10fps",
                                            num_frames=30)

	feature_dir = "./saved_features/test"

	if not os.path.exists(feature_dir):
		os.makedirs(feature_dir)

	all_logits_list = []
	all_features_list = []

	correct = 0
	top1 = AverageMeter()
	top5 = AverageMeter()

	toal_num_video = len(all_frames)
	model.eval()

	for i in range(toal_num_video):

		

		input_data = torch.stack([transform(frame) for frame in all_frames[i]])

		input_var = Variable(input_data.view(-1, 3, input_data.size(2), input_data.size(3)), volatile=True).cuda()

		# 4. extract featrues before the fully connected layer
		features_before_fc = FeatureExtractor(model)

		logits = model(input_var)

		features = features_before_fc(input_var)

		features = features.view(arg.num_frames, 2048, 49)

		logits_np = logits.data.cpu().numpy()

		features_np = np.squeeze(features.data.cpu().numpy())

		
		print("features_np.shape: ", features_np.shape)

		per_video_logits = np.expand_dims(np.mean(logits_np,axis=0), axis=0)
		per_video_label = np.expand_dims(all_labels[i], axis=0)

		per_video_logits = torch.from_numpy(per_video_logits)
		per_video_label  = torch.from_numpy(per_video_label)

		prec1, prec5 = accuracy(per_video_logits, per_video_label, topk=(1, 5))

		top1.update(prec1[0], 1)
		top5.update(prec5[0], 1)

		print('video {} done, total {}/{}, moving Prec@1 {:.3f} Prec@5 {:.3f}'.format(i, i+1,
            toal_num_video,top1.avg, top5.avg))

		if i >= 1500:
			np.save('./saved_features/test/features_{}.npy'.format(i), features_np)
			np.save('./saved_features/test/name_{}.npy'.format(i), all_frame_names[i])
			np.save('./saved_features/test/label_{}.npy'.format(i), per_video_label)
 #    # all_logits = np.asarray(all_logits_list)
 #    # all_frame_names = np.asarray(all_frame_names)
 #    # all_labels = np.asarray(all_labels)
 #    # all_features = np.asarray(all_features_list)

 #    # np.save(os.path.join(feature_dir,"51spa_train_hmdb51_logits.npy"), all_logits)
 #    # np.save(os.path.join(feature_dir,"51spa_train_hmdb51_names.npy"),  all_frame_names)
 #    # np.save(os.path.join(feature_dir,"51spa_train_hmdb51_labels.npy"), all_labels)
 #    # np.save(os.path.join(feature_dir,"51spa_train_hmdb51_features.npy"), all_features)
    

if __name__=='__main__':
    main()

