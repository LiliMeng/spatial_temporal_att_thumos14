'''
Parse Thumos14 temporal annotation from meta file
Author: Lili Meng
Date: Sep 6th, 2018
'''
import numpy as np
import scipy.io as sio
import os

def parse_all_101_val_data():
	val_meta_file='validation_set.mat'
	val_data = sio.loadmat(val_meta_file)['validation_videos'][0]


	all_101_val_video_name = val_data['video_name']
	all_101_val_video_label = val_data['primary_action_index']

	thumos14_101_val_list = open("thumos14_101_val_list.txt", "a")

	for i in range(all_101_val_video_name.shape[0]):
		per_video_name = all_101_val_video_name[i][0]
		per_video_label = all_101_val_video_label[i][0][0]-1

		print("per_video_name: ", per_video_name)
		print("per_video_label: ", per_video_label)
		thumos14_101_val_list.write(per_video_name+' '+str(per_video_label)+'\n')

def parse_20_temporal_val_data():

	temp_20_classes = [6, 8, 11, 20, 21, 22, 23, 25, 30, 32,35, 39, 44, 50, 67, 78, 84, 91, 92, 96]
	val_meta_file='validation_set.mat'
	val_data = sio.loadmat(val_meta_file)['validation_videos'][0]

	all_101_val_video_name = val_data['video_name']
	all_101_val_video_label = val_data['primary_action_index']
	all_101_second_val_video_label = val_data['secondary_actions_indices']

	thumos14_20_val_list = open("thumos14_20_val_only_one_label_list.txt", "a")
	more_than_one_label_list = []
	only_one_label_list = []
	for i in range(all_101_val_video_name.shape[0]):
		per_video_name = all_101_val_video_name[i][0]
		per_video_label = int(all_101_val_video_label[i][0][0]-1)
		per_video_label_second = all_101_second_val_video_label[i]

		#print("per_video_name: ", per_video_name)
		#print("per_video_label: ", per_video_label)
		#print("per_video_label_second: ", per_video_label_second)
		if per_video_label in temp_20_classes:
			print("len(per_video_label_second): ", len(per_video_label_second))
			if len(per_video_label_second)==0:
				only_one_label_list.append(per_video_label)
				
				thumos14_20_val_list.write(per_video_name+' '+str(per_video_label)+'\n')
			else:
				more_than_one_label_list.append(per_video_label_second)

	print("len(more_than_one_label_list): ", len(more_than_one_label_list))
	print("len(only_one_label_list): ", len(only_one_label_list))

def count_num_frames():

	video_dir_list = "thumos14_20_val_only_one_label_list.txt"

	lines = [line.strip() for line in open(video_dir_list).readlines()]

	new_test_list = "new_thumos14_20_val_only_one_label_list.txt"
	new_file_with_img_num = open(new_test_list, "a")

	for line in lines:
		videoname = line.split(' ')[0]
		label = line.split(' ')[1]
		
		video_dir= os.path.join("/media/dataDisk/THUMOS14/THUMOS14_video/thumos14_preprocess/val/frames_10fps", videoname)
		num_files = str(len(os.listdir(video_dir)))

		new_file_with_img_num.write(videoname+" "+label+" "+num_files+"\n")
		
		print("number_files: ", num_files)
	
#parse_20_temporal_val_data()
count_num_frames()