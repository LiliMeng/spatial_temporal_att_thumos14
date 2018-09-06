'''
Parse and make train list for Thumos14 which has already included UCF101
Author: Lili Meng
Date: August 29th, 2018

'''

import numpy as np


def get_train_category():
	category_list = "category_list.txt"
	lines = [line.strip() for line in open(category_list).readlines()]

	category_dict = {}
	for line in lines:
		value = int(line.split(' ')[0])-1
	
		key = line.split(' ')[1]
		
		category_dict[key] = value

	return category_dict

def get_thumos14_label():

	video_dir_list = "thumos14_train.txt"
	lines = [line.strip() for line in open(video_dir_list).readlines()]

	new_train_list = "thumos14_train_with_label.txt"
	new_file_with_img_num = open(new_train_list, "a")
	train_category_dict = get_train_category()


	for line in lines:
		
		
		videoname = line.split(' ')[0]
		label_name = line.split('_')[1]
		print("videoname: ", videoname)
		print("label_name: ", label_name)
		print(train_category_dict[label_name])
		label = train_category_dict[label_name]


		new_file_with_img_num.write(videoname+" "+str(label)+"\n")


get_thumos14_label()
		
		
