import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from skimage import io, color, exposure

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
            path = os.path.join(self.root_dir, "THUMOS14_val_10fps_imgs", video_name)
        elif self.mode == 'test':
            path = os.path.join(self.root_dir, "THUMOS14_test_10fps_imgs", video_name)
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

        if self.mode == 'train':
            
            video_name, start_frame, end_frame, nb_frames = list(self.keys)[idx].split(' ')

            start_frame = int(start_frame)
            nb_frames = int(nb_frames)
            
            clips = []
        
            clips.append(random.randint(start_frame, start_frame+int(nb_frames/3)))
            clips.append(random.randint(start_frame + int(nb_frames/3), start_frame +int(nb_frames*2/3)))
            clips.append(random.randint(start_frame + int(nb_frames*2/3),start_frame + nb_frames))
            
        elif self.mode == 'test':
            video_name, start_frame, end_frame, nb_clips = list(self.keys)[idx].split(' ')
            index = abs(int(nb_clips))
        else:
            raise ValueError('There are only train and val mode')

        label = list(self.values)[idx]
        
        if self.mode=='train':
            data ={}
            for i in range(len(clips)):
                key = 'img'+str(i)
                index = clips[i]
                data[key] = self.load_image(video_name, index)

            sample = (data, label)
        elif self.mode=='test':

            data = self.load_image(video_name,index)
            sample = (video_name, data, label)
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
        for i, line in enumerate(lines):
            videoname = line.split(' ')[0]
            label = int(line.split(' ')[1])
            start_frame= int(float(line.split(' ')[2])*10)
            end_frame = int(float(line.split(' ')[3])*10)
           
            num_imgs = end_frame - start_frame
            
            new_videoname = videoname + " "+str(start_frame) +" "+str(end_frame)
            if split == 'train':
                
                if new_videoname in self.train_video.keys():
                    dup_keys_train.append(new_videoname)
                else:
                    video_path = os.path.join(self.data_path, "THUMOS14_val_10fps_imgs", videoname)
                    self.train_frame_count[new_videoname] = num_imgs
                    self.train_video[new_videoname] = label
            else:
                
                if new_videoname in self.test_video.keys():
                    dup_keys_test.append(new_videoname)
                else:
                    video_path = os.path.join(self.data_path, "THUMOS14_test_10fps_imgs", videoname)
                    self.test_frame_count[new_videoname] = num_imgs
                    self.test_video[new_videoname] = label
        if split == 'train':
            for i in range(len(dup_keys_train)):
                del self.train_video[dup_keys_train[i]]
        elif split == 'test':
            for i in range(len(dup_keys_test)):
                del self.test_video[dup_keys_test[i]]
        else:
            raise Exception("no such split")
       

    def run(self):
        self.load_frame_count('train')
        self.load_frame_count('test')
        self.get_training_dic()
        self.val_sample()
        train_loader = self.train()
        val_loader = self.validate()

        return train_loader, val_loader, self.test_video

    def get_training_dic(self):
       
        self.dic_training={}
        for video in self.train_video:
            
            nb_frame = self.train_frame_count[video]
            key = video+' '+ str(nb_frame)
            self.dic_training[key] = self.train_video[video]
                    
    def val_sample(self):
        print('==> sampling testing frames')
        self.dic_testing={}
        print('test video len:', len(self.test_video))
        for video in self.test_video:
            nb_frame = self.test_frame_count[video]
            interval = int(nb_frame)
            #print("interval: ", interval)
            for i in range(1):
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
        print(training_set[1][0]['img1'].size())

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
    
   dataloader = spatial_dataloader( BATCH_SIZE=4,
                                    num_workers=8,
                                    path='/media/lili/fce9875a-a5c8-4c35-8f60-db60be29ea5d/THUMOS14/THUMOS14_10fps_imgs/',
                                    train_list ='../thumos14_list/new_Thumos_val.txt',
                                    test_list = '../thumos14_list/new_Thumos_test.txt')
   train_loader,val_loader,test_video = dataloader.run()

   for i, (sample, label) in enumerate(train_loader):
        
        print("sample.shape: ", sample)
        print("label: ", label)
        

        break
