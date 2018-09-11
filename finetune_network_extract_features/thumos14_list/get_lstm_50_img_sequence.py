'''
Get LSTM 50 img sequence for temporal annotation
Author: Lili Meng
Date: Sep 9th, 2018
'''

import os
import numpy as np

def load_frames(new_img_list, img_list, video_root_path, num_frames=15):

    new_file_with_start_end_frame = open(new_img_list, "a")

    lines = [line.strip() for line in open(img_list).readlines()]

    frames_for_all_video_list = []
    labels_for_all_video_list = []
    frames_names_for_all_video_list = []

    less_frame_count = 0

    annot_img_index_list = []
    entire_img_index_list = []

    for line in lines:

        video_name = line.split(' ')[0]
        label = int(line.split(' ')[1])
        start_frame = int(float(line.split(' ')[2])*10)
        end_frame = int(float(line.split(' ')[3])*10)
        total_num_imgs_per_video = end_frame-start_frame+1

        video_path = os.path.join(video_root_path, video_name)    

        img_interval = int((total_num_imgs_per_video/num_frames))
        #print("img_interval: ", img_interval)
        #raise Exception("hahha")

        if img_interval != 0:
        	img_index_list = list(range(start_frame, end_frame+1, img_interval))
        else:
        	img_index_list = list(range(start_frame, end_frame+1))

        if total_num_imgs_per_video > num_frames:
            img_index_list = img_index_list[0:num_frames]

        if total_num_imgs_per_video <= num_frames:
            less_frame_count +=1 
            print("total_num_imgs_per_video: ", total_num_imgs_per_video)
            print("less_frame_count: ", less_frame_count)
            final_start_frame = int(start_frame - int(0.5*(50-total_num_imgs_per_video)))
            tmp = final_start_frame 
            final_end_frame = int(end_frame+1 + int(0.5*(50-total_num_imgs_per_video)))
            if final_start_frame < 1:
                final_start_frame =1
                final_end_frame = int(end_frame+1+int(0.5*(50-total_num_imgs_per_video))) +(1-tmp)

            print("old_start_frame: {} final_start_frame: {} ".format(start_frame,final_start_frame))
            print("old_end_frame: {} final_end_frame: {}".format(end_frame, final_end_frame))
            print("total_frame num: {}".format((final_end_frame-final_start_frame)))
            img_index_list_before = list(range(final_start_frame, start_frame))
            img_index_list_after = list(range(end_frame+1, final_end_frame))
            final_img_index_list = img_index_list_before + img_index_list + img_index_list_after
            print("img_index_list_before")
            print(img_index_list_before)
            print("img_index_list")
            print(img_index_list)
            print("img_index_list_after")
            print(img_index_list_after)
            print("final_img_index_list")

            print(final_img_index_list)
            print("len(final_img_index_list): ", len(final_img_index_list))
            
        else:
            final_start_frame = start_frame-10*img_interval
            final_end_frame = end_frame + 10*img_interval
            tmp = final_start_frame
            if final_start_frame < 1:
                final_start_frame =1
                final_end_frame = end_frame + 10*img_interval +(1-tmp)

            print("old_start_frame: {} final_start_frame: {} ".format(start_frame,final_start_frame))
            print("old_end_frame: {} final_end_frame: {}".format(end_frame, final_end_frame))
            print("total_frame num: {}".format((final_end_frame-final_start_frame)))
            img_index_list_before = list(range(final_start_frame, start_frame, img_interval))
            img_index_list_after = list(range(end_frame+1, final_end_frame+1, img_interval))
            final_img_index_list = img_index_list_before + img_index_list + img_index_list_after
            print("img_index_list_before")
            print(img_index_list_before)
            print("img_index_list")
            print(img_index_list)
            print("img_index_list_after")
            print(img_index_list_after)
            print("final_img_index_list")

            print(final_img_index_list)
            print("len(final_img_index_list): ", len(final_img_index_list))
        
        print("len(img_index_list_before): ", len(img_index_list_before))
        print("len(img_index_list_after): ", len(img_index_list_after))
        print("len(img_index_list): ", len(img_index_list))
        if len(final_img_index_list)< 50:
            appending_after_list = list(range(final_end_frame+1, final_end_frame+1+50-len(final_img_index_list)))
            final_img_index_list += appending_after_list
        assert(len(final_img_index_list)==50)
        #new_file_with_start_end_frame.write(video_name+' '+str(label)+' '+str(final_start_frame)+' '+str(final_end_frame)+'\n')
        annot_img_index_list.append(img_index_list)
        entire_img_index_list.append(final_img_index_list)

    return annot_img_index_list, entire_img_index_list

load_frames(new_img_list = "new_file_with_start_end_frame_val.txt",
            img_list = "final_thumos_14_20_one_label_temporal_val.txt",
            video_root_path = "/media/dataDisk/THUMOS14/THUMOS14_video/thumos14_preprocess/val/frames_10fps",
            num_frames=30)