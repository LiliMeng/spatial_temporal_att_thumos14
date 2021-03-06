import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from skimage import io, color, exposure
from pathlib import Path



class spatial_dataset(Dataset):  
    def __init__(self, dic, root_dir, mode, transform=None):
 
        self.keys = dic.keys()
        #print(dic.keys())
        self.values = dic.values()
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def load_image(self, video_name, index):

        if self.mode == 'train':
            path = os.path.join(self.root_dir, "val/frames_10fps", video_name)

        elif self.mode == 'test':
            path = os.path.join(self.root_dir, "test/frames_10fps", video_name)
            
        else:
            raise Exception("No such mode")

        img = Image.open(os.path.join(path,  str('%05d'%(index)) + '.jpg'))
        try:
            transformed_img = self.transform(img)
        except:
            print(os.path.join(path,  str('%05d'%(index)) + '.jpg'))
            img.close()

        return transformed_img

    def __getitem__(self, idx):
        
        
        label = list(self.values)[idx]
        video_name, start_frame, end_frame, nb_frames = list(self.keys)[idx].split(' ')

        if self.mode == 'train':

            start_frame = int(start_frame)
            end_frame = int(end_frame)
            nb_frames = int(nb_frames)

            clips = []

            #if start_frame ==0:
            start_frame += 1
            #end_frame += 1
            
            clips.append(random.randint(start_frame, start_frame+int(nb_frames/5)))
            clips.append(random.randint(start_frame + int(nb_frames/5), start_frame +int(nb_frames*2/5)))
            clips.append(random.randint(start_frame + int(nb_frames*2/5), start_frame +int(nb_frames*3/5)))
            clips.append(random.randint(start_frame + int(nb_frames*3/5), start_frame +int(nb_frames*4/5)))
            clips.append(random.randint(start_frame + int(nb_frames*4/5),int(end_frame)))

            data ={}
            for i in range(len(clips)):
                key = 'img'+str(i)
                index = clips[i]
                data[key] = self.load_image(video_name, index)

            sample = (data, label)
            
                          
        elif self.mode == 'test':
            
            index = abs(int(nb_frames))
            data = self.load_image(video_name,index)
            new_videoname = video_name + " " + start_frame + " "+end_frame
            sample = (new_videoname, data, label)
        else:
            raise ValueError('There are only train and val mode')
           
        return sample

class spatial_dataloader():
    def __init__(self, BATCH_SIZE, num_workers, path, train_list, test_list):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers=num_workers
        self.data_path=path
        self.train_frame_count ={}
        self.test_frame_count = {}
        self.train_video = {}
        self.test_video = {}
        self.train_list = train_list
        self.test_list = test_list
    

    def load_frame_count(self, split):
      
        if split == 'train':
        	split_list = self.train_list
        else:
        	split_list = self.test_list
        lines = [line.strip() for line in open(split_list).readlines()]
        
        dup_keys_train = []
        dup_keys_test = []
        not_exist_frame_video_train = []
        not_exist_frame_video_test = []
        for i, line in enumerate(lines):
            videoname = line.split(' ')[0]
            label = int(line.split(' ')[1])
            start_frame= int(float(line.split(' ')[2])*10)
            end_frame = int(float(line.split(' ')[3])*10)
           
            num_imgs = end_frame - start_frame
            
            new_videoname = videoname + " "+str(start_frame) +" "+str(end_frame)
            
            if split == 'train':   
                last_frame_name_train = os.path.join(self.data_path, "val/frames_10fps", videoname, str('%05d'%(end_frame)) + '.jpg')
                if not Path(last_frame_name_train).is_file():
                    not_exist_frame_video_train.append(new_videoname)
                    print("no such file exist in train: ", new_videoname)
                if new_videoname in self.train_video.keys():
                    dup_keys_train.append(new_videoname)
                else:
                    video_path = os.path.join(self.data_path, "val/frames_10fps", videoname)
                    self.train_frame_count[new_videoname] = num_imgs
                    self.train_video[new_videoname] = label
            elif split == 'test': 
                last_frame_name_test = os.path.join(self.data_path, "test/frames_10fps", videoname, str('%05d'%(end_frame)) + '.jpg')
                if not Path(last_frame_name_test).is_file():
                    print("no such file exist in the test: ", new_videoname)
                    not_exist_frame_video_test.append(new_videoname)
                if new_videoname in self.test_video.keys():
                    dup_keys_test.append(new_videoname)
                else:
                    video_path = os.path.join(self.data_path, "test/frames_10fps", videoname)
                    self.test_frame_count[new_videoname] = num_imgs
                    self.test_video[new_videoname] = label
            else:
                raise Exception("no such mode exist, only support train and test")

        if split == 'train':
            for i in range(len(not_exist_frame_video_train)):
                del self.train_video[not_exist_frame_video_train[i]]
            for i in range(len(dup_keys_train)):
                del self.train_video[dup_keys_train[i]]
        elif split == 'test':
            for i in range(len(not_exist_frame_video_test)):
                del self.test_video[not_exist_frame_video_test[i]]
            for i in range(len(dup_keys_test)):
                del self.test_video[dup_keys_test[i]]
        else:
            raise Exception("no such split, only train and test mode is provided")
       

    def run(self):
        self.load_frame_count('train')
        self.load_frame_count('test')
        self.get_training_dic()
        self.val_sample(num_frame_per_video=6)
        train_loader = self.train()
        val_loader = self.validate()

        return train_loader, val_loader, self.test_video

    def get_training_dic(self):
       
        self.dic_training={}
        for video in self.train_video:
            
            nb_frame = self.train_frame_count[video]
            key = video+' '+ str(nb_frame)
            self.dic_training[key] = self.train_video[video]
                    
    def val_sample(self, num_frame_per_video):
        print('==> sampling testing frames')
        self.dic_testing={}
        print('test video len:', len(self.test_video))
        for video in self.test_video:
            nb_frame = self.test_frame_count[video]-num_frame_per_video+1
            
            interval = int(nb_frame/num_frame_per_video)
           
            for i in range(num_frame_per_video):
                frame = i*interval
                key = video+ ' '+str(frame+1)
                self.dic_testing[key] = self.test_video[video]      

    def train(self):
        training_set = spatial_dataset(dic=self.dic_training, root_dir=self.data_path, mode='train', transform = transforms.Compose([
                transforms.Scale([256,256]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        print('==> Training data :',len(training_set),'frames')

        #print(training_set[1][0]['img1'].size())

        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers)
            
        return train_loader

    def validate(self):
        validation_set = spatial_dataset(dic=self.dic_testing, root_dir=self.data_path, mode='test', transform = transforms.Compose([
                #transforms.Scale([256,256]),
                #transforms.CenterCrop(224),
                transforms.Scale([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        
        print('==> Validation data :',len(validation_set),'frames')
        print(validation_set[1][1].size())

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)
        return val_loader


if __name__ == '__main__':
    
   data_loader = spatial_dataloader(
                        BATCH_SIZE=16,
                        num_workers=8,
                        path='/media/dataDisk/THUMOS14/THUMOS14_video/thumos14_preprocess/',
                        train_list ='../thumos14_list/new_thumos_14_20_one_label_temporal_val.txt',
                        test_list = '../thumos14_list/new_thumos_14_20_one_label_temporal_test.txt')
   train_loader,val_loader,test_video = data_loader.run()


   
   for i, (data, label) in enumerate(train_loader):
        print("train i", i)

   for i, (keys, data, label) in enumerate(val_loader):
        
        print("test i", i)
        

   
