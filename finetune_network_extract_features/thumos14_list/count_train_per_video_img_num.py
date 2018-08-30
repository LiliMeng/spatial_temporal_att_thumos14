'''
Count UCF101 train image number per video

Author: Lili Meng
Date: August 29th, 2018

'''
import os
import numpy as np

video_dir_list = "thumos14_train_with_label.txt"

lines = [line.strip() for line in open(video_dir_list).readlines()]

new_train_list = "new_Thumos_train.txt"
new_file_with_img_num = open(new_train_list, "a")

for line in lines:
	videoname = line.split(' ')[0]
	label = line.split(' ')[1]
	
	videoname = os.path.join("/media/dataDisk/THUMOS14/THUMOS_14_training/UCF101", videoname)
	num_files = str(len(os.listdir(videoname)))

	new_file_with_img_num.write(videoname+" "+label+" "+num_files+"\n")
	
	print("number_files: ", num_files)
	