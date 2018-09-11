'''
Count UCF101 train image number per video

Author: Lili Meng
Date: August 29th, 2018

'''
import os
import numpy as np

video_dir_list = "final_thumos_14_20_one_label_temporal_test.txt"

lines = [line.strip() for line in open(video_dir_list).readlines()]

new_train_list = "final_thumos_14_20_one_label_temporal_test_with_img_num.txt"
new_file_with_img_num = open(new_train_list, "a")

for line in lines:
	videoname = line.split(' ')[0]
	label = line.split(' ')[1]
	start_time = line.split(' ')[2]
	end_time = line.split(' ')[3]
	
	video_dir= os.path.join("/media/dataDisk/THUMOS14/THUMOS14_video/thumos14_preprocess/test/frames_10fps", videoname)
	num_files = str(len(os.listdir(video_dir)))

	new_file_with_img_num.write(videoname+" "+label+" "+start_time+" "+end_time+" "+num_files+"\n")
	
	print("number_files: ", num_files)
	