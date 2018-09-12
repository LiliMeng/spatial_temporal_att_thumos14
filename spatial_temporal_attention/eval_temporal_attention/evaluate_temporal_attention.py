'''
Evaluate thumos 14 temporal attention results
Author: Lili Meng (menglili@cs.ubc.ca)
Date: Sep 11th, 2018
'''

import numpy as np
import os

att_weights_dir = "../saved_weights/Contrast_0.0001_TV_reg1e-05_mask_LRPatience5_Adam0.0001_decay0.0001_dropout_0.2_Temporal_ConvLSTM_hidden512_regFactor_1_Sep_12_11_00"

att_weights_file = os.path.join(att_weights_dir, "test_att_weights.npy")
test_name_file = os.path.join(att_weights_dir, "test_name.npy")
attention_weights = np.load(att_weights_file)

test_img_names = np.load(test_name_file)
test_img_names = np.concatenate(test_img_names, axis=0)


annotated_img_list = np.load("/media/dataDisk/Video/spatial_temporal_att_thumos14/finetune_network_extract_features/thumos14_list/test_img_index_list.npy")

IOU_05_count = 0
for i in range(attention_weights.shape[0]):
	print("video_name: ", test_img_names[i][0].split('/')[-2])
	good_img_index = []
	all_img_index = []
	for j in range(attention_weights.shape[1]):
		img_index = int(test_img_names[i][j].split('/')[-1].split('.jpg')[0])
		all_img_index.append(img_index)
		if attention_weights[i][j] > 0.025:
			good_img_index.append(img_index)

	print("len(good_img_index): ", len(good_img_index))
	count_in_tmp_annot_num =0
	for j in range(len(good_img_index)):
		index = good_img_index[j]
		if index in annotated_img_list[i]:
			count_in_tmp_annot_num+=1

	print("count_in_tmp_annot_num: ", count_in_tmp_annot_num)	

	IOU = count_in_tmp_annot_num/30
	print("IOU: ", IOU)

	if IOU > 0.1:
		IOU_05_count+=1

	
print("the number of IOU is larger than 0.5 is: ", IOU_05_count)
print("the percentage of IOU is larger than 0.5 is: ", IOU_05_count/attention_weights.shape[0])
