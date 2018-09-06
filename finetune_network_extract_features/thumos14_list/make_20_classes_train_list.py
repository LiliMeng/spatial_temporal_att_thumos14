'''
Make train list for Thumos 14 20 classes temporal annotation

Author: Lili Meng
Date: Sep 6th, 2018
'''

import numpy as np

temp_20_classes = [6, 8, 11, 20, 21, 22, 23, 25, 30, 32,35, 39, 44, 50, 67, 78, 84, 91, 92, 96]


all_101_classes_train_list = "new_Thumos_train.txt"

lines = [line.strip() for line in open(all_101_classes_train_list).readlines()]

temporal_20_classes_train_list = open("temporal_Thumos_20classes_train.txt", "a")

for line in lines:
	video_name = line.split(' ')[0]
	video_label = line.split(' ')[1]
	video_num = line.split(' ')[2]

	print("video_name: ", video_name)
	print("video_label: ", video_label)
	if int(video_label) in temp_20_classes:
		print("video_label ", video_label)
		temporal_20_classes_train_list.write(video_name+' '+video_label+' '+video_num+'\n')


