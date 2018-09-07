'''
Parse Thumos14 temporal annotation from meta file
Author: Lili Meng
Date: Sep 6th, 2018
'''
import numpy as np
import scipy.io as sio


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

	thumos14_20_val_list = open("thumos14_20_val_list.txt", "a")

	for i in range(all_101_val_video_name.shape[0]):
		per_video_name = all_101_val_video_name[i][0]
		per_video_label = int(all_101_val_video_label[i][0][0]-1)

		print("per_video_name: ", per_video_name)
		print("per_video_label: ", per_video_label)
		if per_video_label in temp_20_classes:
			thumos14_20_val_list.write(per_video_name+' '+str(per_video_label)+'\n')

parse_20_temporal_val_data()
