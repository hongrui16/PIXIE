import json
import os, cv2, sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from . import detectors
from utils.image_process import crop_resize_image, crop_resize_image_batch, pad_resize_image



class OpenPoseDataset(Dataset):
    def __init__(self, data_dir, split = None, crop_size=224, hd_size = 512, logger = None):
        '''
            testpath: folder, imagepath_list, image path, video path
        '''
        self.data_dir = data_dir
        self.split = split
        self.crop_size = crop_size
        self.hd_size = hd_size
        self.logger = logger

        if split is None:
            self.img_dir = os.path.join(data_dir, 'images')
            self.keypoint_dir = os.path.join(data_dir, 'keypoints')
        else:
            self.img_dir = os.path.join(data_dir, split, 'images')
            self.keypoint_dir = os.path.join(data_dir, split, 'keypoints')
        
        if os.path.isdir(self.img_dir):
            self.imagepath_list = glob(self.img_dir + '/*.jpg') +  glob(self.img_dir + '/*.png') + glob(self.img_dir + '/*.jpeg')
        else:
            self.logger.info(f'please check the input dir: {self.img_dir}')
            exit()
        
        self.logger.info(f'{self.img_dir} has {len(self.imagepath_list)} images')
        # self.imagepath_list = sorted(self.imagepath_list)


    def __len__(self):
        return len(self.imagepath_list)

    def __getitem__(self, index):
        # print('index:', index, type(index))
        img_path = self.imagepath_list[index]
        # img_name = img_path.split('/')[-1].split('.')[0]
        img_name = os.path.basename(img_path).split('.')[0]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        ori_h, ori_w, _ = image.shape
        image, scale, start_x, start_y = pad_resize_image(image, self.hd_size)
        
        # convert image to tensor
        image = image.transpose(2,0,1)
        image = torch.tensor(image).float()/255.0
        

        # get keypoints
        keypoint_path = os.path.join(self.keypoint_dir, img_name + '_keypoints.json')
        with open(keypoint_path) as f:
            data = json.load(f)
            for person in data['people']:
                pose_kpts = person['pose_keypoints_2d']
                # print('keypoints:', keypoints)
                hand_left_kpts = person['hand_left_keypoints_2d']
                hand_right_kpts = person['hand_right_keypoints_2d']
                face_kpts = person['face_keypoints_2d']

                ## convert to array
                pose_kpts = np.array(pose_kpts).reshape(-1, 3)
                hand_left_kpts = np.array(hand_left_kpts).reshape(-1, 3)
                hand_right_kpts = np.array(hand_right_kpts).reshape(-1, 3)
                face_kpts = np.array(face_kpts).reshape(-1, 3)
            
        # update keypoints
        pose_kpts[:, 0] = (pose_kpts[:, 0] + start_x) * scale
        pose_kpts[:, 1] = (pose_kpts[:, 1] + start_y) * scale
        hand_left_kpts[:, 0] = (hand_left_kpts[:, 0] + start_x) * scale
        hand_left_kpts[:, 1] = (hand_left_kpts[:, 1] + start_y) * scale
        hand_right_kpts[:, 0] = (hand_right_kpts[:, 0] + start_x) * scale
        hand_right_kpts[:, 1] = (hand_right_kpts[:, 1] + start_y) * scale
        face_kpts[:, 0] = (face_kpts[:, 0] + start_x) * scale
        face_kpts[:, 1] = (face_kpts[:, 1] + start_y) * scale
        

        # crop image
        return {'image': image,
                'name': img_name,
                'imagepath': img_path,
                # 'image_hd': None,
                'pose_keypoints_2d': torch.tensor(pose_kpts).float(),
                'hand_left_keypoints_2d': torch.tensor(hand_left_kpts).float(),
                'hand_right_keypoints_2d': torch.tensor(hand_right_kpts).float(),
                'face_keypoints_2d': torch.tensor(face_kpts).float(),
                }


if __name__ == "__main__":
    testpath = r'C:\Users\hongr\Documents\GMU_research\computerVersion\hand_modeling\PIXIE\TestSamples\body\pexels-andrea-piacquadio-972937.jpg'

    