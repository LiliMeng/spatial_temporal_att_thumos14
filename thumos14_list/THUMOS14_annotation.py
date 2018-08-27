'''
Convert to the train and validation list used for training and testing
from THUMOS14 temporal annotation files
Author: Lili Meng (menglili@cs.ubc.ca)
Date: August 26th, 2018
'''

import numpy as np
import os

def make_new_list(img_list, new_img_list, class_label):
	
	new_file_with_img_num = open(new_img_list, "a")
	lines = [line.strip() for line in open(img_list).readlines()]

	for line in lines:
	    videoname = line.split(' ')[0]
	    blank = line.split(' ')[1]
	    start_time = line.split(' ')[2]
	    end_time = line.split(' ')[3]
	    label = str(class_label)

	    print("videoname: ", videoname)
	    print("start_time: ", start_time)
	    print("end_time: ", end_time)
	    new_file_with_img_num.write(videoname+" "+label+" "+start_time +" "+end_time+"\n") 

	new_file_with_img_num.close()

def main():

	list_folder_path = "/media/lili/fce9875a-a5c8-4c35-8f60-db60be29ea5d/THUMOS14/TH14_Temporal_Annotations_Test/annotations/annotation/"
	new_val_img_list = "./new_Thumos_test.txt"

	label =0
	for val_img_name in sorted(os.listdir(list_folder_path)):
		val_img_folder = os.path.join(list_folder_path, val_img_name)
		make_new_list(val_img_folder, new_val_img_list, label)
		label+=1

if __name__== "__main__":
	main()