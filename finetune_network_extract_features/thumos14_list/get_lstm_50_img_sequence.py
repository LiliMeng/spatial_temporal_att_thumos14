'''
Get LSTM 50 img sequence for temporal annotation
Author: Lili Meng
Date: Sep 9th, 2018
'''

import os
import numpy as np

def load_frames(new_img_list, img_list, video_root_path, num_frames=15):

    lines = [line.strip() for line in open(img_list).readlines()]

    frames_for_all_video_list = []
    labels_for_all_video_list = []
    frames_names_for_all_video_list = []

    less_frame_count = 0
    annot_img_index_list = []
    entire_img_index_list = []

    count=0
    for line in lines:
        count+=1
        video_name = line.split(' ')[0]
        label = int(line.split(' ')[1])
        start_frame = int(float(line.split(' ')[2])*10)
        end_frame = int(float(line.split(' ')[3])*10)
        num_frames_entire_video = int(line.split(' ')[4])
        total_num_imgs_per_video = end_frame-start_frame

        video_path = os.path.join(video_root_path, video_name)

        
        img_interval = int((total_num_imgs_per_video/num_frames))
        
        if img_interval != 0:
            img_index_list = list(range(start_frame, end_frame+1, img_interval))
        else:
            img_index_list = list(range(start_frame, end_frame+1))

        if total_num_imgs_per_video > num_frames:
            img_index_list = img_index_list[0:num_frames]
            
        if total_num_imgs_per_video <= num_frames:
            less_frame_count +=1 
        
            final_start_frame = int(start_frame - int(0.5*(50-total_num_imgs_per_video)))
            final_end_frame = int(end_frame+1 + int(0.5*(50-total_num_imgs_per_video)))
            
            if final_start_frame < 1:
                final_start_frame =1
             #   final_end_frame = int(end_frame+1+int(0.5*(50-total_num_imgs_per_video))) +(1-tmp)

            img_index_list_before = list(range(final_start_frame, start_frame))
            img_index_list_after = list(range(end_frame+1, final_end_frame))
            final_img_index_list = img_index_list_before + img_index_list + img_index_list_after
            
        else:
            final_start_frame = start_frame-10*img_interval
            final_end_frame = end_frame + 10*img_interval

            tmp = final_start_frame
            if final_start_frame < 1:
                final_start_frame =1
                #final_end_frame = end_frame + 10*img_interval +(1-tmp)

            img_index_list_before = list(range(final_start_frame, start_frame, img_interval))
            img_index_list_after = list(range(end_frame+1, final_end_frame+1, img_interval))

          
            if final_end_frame > num_frames_entire_video:

                final_end_frame = end_frame +10

                if final_end_frame > num_frames_entire_video:
                    img_index_list_after = list(range(end_frame+1, num_frames_entire_video))
                else:
                    img_index_list_after = list(range(end_frame+1, final_end_frame+10))

            final_img_index_list = img_index_list_before + img_index_list + img_index_list_after

        if len(final_img_index_list)< 50:
            appending_after_list = list(range(final_end_frame+1, final_end_frame+1+50-len(final_img_index_list)))
            for i in range(len(appending_after_list)):
                if appending_after_list[i] > num_frames_entire_video:
                    appending_after_list[i] = num_frames_entire_video

            final_img_index_list += appending_after_list
        else:
            final_img_index_list = final_img_index_list[0:50]



        print("i: ", count)
        assert(len(final_img_index_list)==50)

        annot_img_index_list.append(img_index_list)
        
    np.save("test_img_index_list.npy", np.asarray(annot_img_index_list))
        # imgs_per_video_list = []
        # imgs_names_per_video_list = []
        # label_per_video_list = []
        # exist_frame_count = 0

    
        # for i in range(0, len(final_img_index_list)):

        #     img_name = os.path.join(video_path,  str('%05d'%(final_img_index_list[i])) + '.jpg')

        #     if os.path.isfile(img_name):
        #         exist_frame_count +=1
        #     else:
        #         img_name = os.path.join(video_path, str('%05d'%(final_img_index_list[exist_frame_count-1])) + '.jpg')

            
        #     img = Image.open(img_name).convert('RGB')

        #     try:
        #         imgs_per_video_list.append(img)
        #         imgs_names_per_video_list.append(img_name) 
        #     except:
        #         print(os.path.join(path, str('%05d'%(index)) + '.jpg'))
        #         img.close()

        #frames_for_all_video_list.append(imgs_per_video_list)
        #labels_for_all_video_list.append(label)
        #frames_names_for_all_video_list.append(imgs_names_per_video_list)

    return annot_img_index_list #frames_for_all_video_list, frames_names_for_all_video_list, labels_for_all_video_list


load_frames(new_img_list = "new_file_with_start_end_frame_test.txt",
            img_list = "final_thumos_14_20_one_label_temporal_test_with_img_num.txt",
            video_root_path = "/media/dataDisk/THUMOS14/THUMOS14_video/thumos14_preprocess/test/frames_10fps",
            num_frames=30)
