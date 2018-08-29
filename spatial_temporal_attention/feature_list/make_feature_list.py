'''
Make feature list for the csv file needed by the feature_loader.py

Author: Lili Meng
Date: August 29th, 2018

'''

import numpy as np
import os


feature_dir = "../../../spa_features/test/features/"
txt_file = open("feature_test_list.txt", mode='a')

txt_file.write("Feature"+"\n")


for feature_filename in sorted(os.listdir(feature_dir)):
	
	if '.npy' in feature_filename:
		print("feature_filename: ", feature_filename)

		txt_file.write(feature_filename+"\n")