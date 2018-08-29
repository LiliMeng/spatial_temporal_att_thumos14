'''
make thumos14 list for ResNet spatial CNN training

Author: Lili Meng
Date: August 26th, 2018
'''

import numpy as np
import os

def class_mapping_dict():
	mapping_list = "thumos_class_mapping.txt"
	lines = [line.strip() for line in open(mapping_list).readlines()]

	mapping_dict = {}
	for line in lines:
		old_key = int(line.split(' ')[0])
		new_key = int(line.split(' ')[1])
		mapping_dict[old_key] = new_key

	return mapping_dict

def make_new_list(img_list, new_img_list, class_label, mapping_dict):
	
	new_file_with_img_num = open(new_img_list, "a")
	lines = [line.strip() for line in open(img_list).readlines()]

	for line in lines:
	    videoname = line.split(' ')[0]
	    blank = line.split(' ')[1]
	    start_time = line.split(' ')[2]
	    end_time = line.split(' ')[3]
	    label = str(mapping_dict[class_label]-1)

	    print("videoname: ", videoname)
	    print("start_time: ", start_time)
	    print("end_time: ", end_time)
	    new_file_with_img_num.write(videoname+" "+label+" "+start_time +" "+end_time+"\n") 

	new_file_with_img_num.close()

def main():

	mapping_dict = class_mapping_dict()
	print("mapping_dict")
	print(mapping_dict)

	list_folder_path = "/media/dataDisk/THUMOS14/TH14_Temporal_Annotations_Test/annotations/annotation/"
	new_val_img_list = "./new_Thumos_test.txt"

	label =1
	for val_img_name in sorted(os.listdir(list_folder_path)):
		val_img_folder = os.path.join(list_folder_path, val_img_name)
		make_new_list(val_img_folder, new_val_img_list, label, mapping_dict)
		label+=1

if __name__== "__main__":
	main()