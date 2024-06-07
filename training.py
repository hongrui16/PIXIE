import os, sys
import numpy as np
import torch.backends.cudnn as cudnn
import torch
from tqdm import tqdm
import argparse
import cv2
import open3d as o3d
import keyboard  # 导入keyboard库
import plotly.graph_objects as go
from datetime import datetime
from PIL import Image
from torch.nn import functional as F
# logging
import logging
import torch.optim as optim



sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pixielib.pixieTorch import PIXIE
# from pixielib.pixie_parallel import PIXIE
from pixielib.visualizer import Visualizer
from pixielib.datasets.body_datasets import TestData
from pixielib.utils import util
from pixielib.utils.config import cfg as pixie_cfg
from pixielib.utils.tensor_cropper import transform_points
from pixielib.models.smplx_openpose_joints import SMPLX_names, selected_mapping, openpose_in_smplx_mapping
from pixielib.datasets.build_dataloader import build_dataloader
from pixielib.datasets import detectors
from pixielib.utils.visualizer import vis_mesh_points, plotly_save_point_cloud
from pixielib.criterion.compute_loss import ComputeL2Loss

from pixielib.utils.image_process import crop_resize_image_batch, get_bboxes_from_keypoints, compute_mean_xy



class Worker:
    def __init__(self, args = None) -> None:        
        args.data_dir = r'C:\Users\hongr\Documents\GMU_research\computerVersion\hand_modeling\smile_data\color_openpose\images'
        args.data_dir = r'C:\Users\hongr\Documents\GMU_research\computerVersion\hand_modeling\expose\samples'
        args.data_dir = r'C:\Users\hongr\Documents\GMU_research\computerVersion\hand_modeling\signLangWord\WLASL_examples'
        args.data_dir = r'TestSamples\body\singleMan'
        args.data_dir = r'TestSamples\body\singleMan1image'
        # args.data_dir = 'test_image'
        
        self.input_data_dir = args.data_dir
        log_dir = args.log_dir
        batch_size = args.batch_size
        self.fast_train = args.fast_train
        self.max_epoch = args.max_epoch
        debug = args.debug
        self.compute_2d_loss = args.compute_2d_loss
        self.compute_3d_loss = args.compute_3d_loss

        self.part_count_dict = {'pose': 25}#, 'hand': 42, 'face': 70}
        self.dataset_type = 'openpose'
        self.detector_name = 'rcnn'
        self.input_size = 224
        self.hd_size = 512
        self.scale = 1.2
        
        self.debug = False or debug
        self.vis = False
        self.dump_img = True
        
        if not self.debug:
            batch_size = batch_size
            if self.vis:
                self.mesh_point_visualizer = vis_mesh_points()
        else:
            batch_size = 2

        log_dir = os.path.join(log_dir, "{}_{:%Y-%m-%d_%H-%M-%S}".format(self.dataset_type, datetime.now()))
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.dump_img_dir = os.path.join(log_dir, 'dump_img')
        os.makedirs(self.dump_img_dir, exist_ok=True)
        
        
        
        # Log to file & tensorboard writer
        self.log_path = os.path.join(log_dir, "info.log")
        logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(self.log_path), logging.StreamHandler()])
        self.logger = logging.getLogger('PIXIE')
        self.logger.info(f'log_dir: {log_dir}')

        ## print all the args into log file
        logging.info(f"\n***************hyperparameters***********************************************")
        kwargs = vars(args)
        for key, value in kwargs.items():
            self.logger.info(f"{key}: {value}")
        
        
        # check env
        if not torch.cuda.is_available():
            self.logger.info('CUDA is not available! use CPU instead')
            self.device = 'cpu'
        else:
            cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.enabled = True
            self.device = 'cuda:0'

        logging.info(f"\n*******************************build dataset*******************************")
        ### build dataset
        self.tr_dataset, self.tr_dataloader = build_dataloader(self.input_data_dir, self.dataset_type, 
                                                 batch_size = batch_size, device=self.device, logger = self.logger,
                                                 args = args)
        
        
        logging.info(f"\n*******************************build PIXIE model*******************************")
        #-- run PIXIE
        pixie_cfg.model.use_tex = args.useTex
        self.model = PIXIE(config = pixie_cfg, device=self.device, logger=self.logger, 
                           input_size = self.input_size, project_type='perspective')



        logging.info(f"\n*******************************build detector*******************************")
        if self.detector_name == 'rcnn':
            self.detector = detectors.FasterRCNN(device=self.device)
        elif self.detector_name == 'keypoint':
            self.detector = detectors.KeypointRCNN(device=self.device)
        else:
            self.logger.info('no detector is used')
            exit()
        
        self.criteria = ComputeL2Loss()
        
        logging.info(f"\n*******************************build optimizer*******************************")        
        patience = 10
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=patience//3)
        
        logging.info(f"\n*******************************resume and finetune*******************************")

        self.start_epoch = 0
        self.best_epoch_loss = float('inf')
        if args.resume is not None:
            # load model from checkpoint
            checkpoint = torch.load(args.resume)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f'resume model from {args.resume}; fine_tune: {args.fine_tune}')
            if not args.fine_tune:                
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])    
                if 'best_epoch_loss' in checkpoint:
                    self.best_epoch_loss = checkpoint['best_epoch_loss']
                if 'epoch' in checkpoint:
                    self.start_epoch = checkpoint['epoch'] + 1
        
        logging.info(f"\n*******************************load openpose and smplx mapping*******************************")
        self.openpose_in_smplx_mapping = openpose_in_smplx_mapping
        
        

        self.logger.info(f"******************* Init finished! *******************")

    def train_epoch(self, epoch):
        points_confidence_threshold = 0.1
        split = 'train'
        tbar = tqdm(self.tr_dataloader)
        width = 6  # Total width including the string length
        formatted_split = split.rjust(width)

        self.model.train()
        loss_epoch = []
        for idx, batch in enumerate(tbar):
            self.optimizer.zero_grad()
            
            image = batch['image'].to(self.device)
            image_name_prefix = batch['name']

            gt_pose_keypoints_2d = batch['pose_keypoints_2d'].to(self.device)
            gt_hand_left_keypoints_2d = batch['hand_left_keypoints_2d'].to(self.device)
            gt_hand_right_keypoints_2d = batch['hand_right_keypoints_2d'].to(self.device)
            gt_face_keypoints_2d = batch['face_keypoints_2d'].to(self.device)

            if 'joints_3d' in batch:
                gt_joints_3d = batch['joints_3d'].to(self.device)
            
            bboxes = self.detector.run_batch(image)
            if isinstance(bboxes, np.ndarray):
                bboxes = torch.tensor(bboxes).float()
            # 将 bboxes 移动到所需设备，并确保它们不启用梯度跟踪
            bboxes = bboxes.to(self.device).detach()
            
            # self.draw_kpts_on_image(image, image_name_prefix, epoch, gt_pose_keypoints_2d, part = 'pose', stage = '1', gt_bboxes = bboxes)
            
            bs, c, h, w = image.shape
            ## expand the bboxes by self.scale
            bboxes[:, 0] = torch.clamp(bboxes[:, 0] - (self.scale - 1) * (bboxes[:, 2] - bboxes[:, 0]) / 2, 0, w-1)
            bboxes[:, 1] = torch.clamp(bboxes[:, 1] - (self.scale - 1) * (bboxes[:, 3] - bboxes[:, 1]) / 2, 0, h-1)
            bboxes[:, 2] = torch.clamp(bboxes[:, 2] + (self.scale - 1) * (bboxes[:, 2] - bboxes[:, 0]) / 2, 0, w-1)
            bboxes[:, 3] = torch.clamp(bboxes[:, 3] + (self.scale - 1) * (bboxes[:, 3] - bboxes[:, 1]) / 2, 0, h-1)

            ### deduce the keypoints by the xmin, ymin of bboxes, because of croping the image based on bboxes
            # print('gt_pose_keypoints_2d:', gt_pose_keypoints_2d.shape) # torch.Size([bs, 25, 3])
            # print('bboxes:', bboxes.shape) # torch.Size([bs, 4])

            bboxes_minxy = bboxes[:, :2].unsqueeze(1)
            gt_pose_keypoints_2d[:, :, :2] -= bboxes_minxy
            gt_hand_left_keypoints_2d[:, :, :2] -= bboxes_minxy
            gt_hand_right_keypoints_2d[:, :, :2] -= bboxes_minxy
            gt_face_keypoints_2d[:, :, :2] -= bboxes_minxy
            cropped_images_hd, scales, left_pad, top_pad = crop_resize_image_batch(image, bboxes, self.hd_size)
            # print('cropped_images_hd:', cropped_images_hd.shape) # torch.Size([bs, 3, 512, 512])
            # print('scales_x:', scales_x.shape) # torch.Size([bs])
            # print('scales_y:', scales_y.shape) # torch.Size([bs])
            # print('left_pad:', left_pad.shape) # torch.Size([bs])
            # print('top_pad:', top_pad.shape) # torch.Size([bs])
            # print('gt_pose_keypoints_2d:', gt_pose_keypoints_2d.shape) # torch.Size([bs, 25, 3])
            # print('gt_hand_left_keypoints_2d:', gt_hand_left_keypoints_2d.shape) # torch.Size([bs, 21, 3])

            # 扩展 left_pad 和 top_pad 的形状到 [bs, 1]
            left_pad_expanded = left_pad.view(-1, 1)  # [bs, 1]
            top_pad_expanded = top_pad.view(-1, 1)  # [bs, ]

            gt_pose_keypoints_2d[:, :, 0] += left_pad_expanded
            gt_pose_keypoints_2d[:, :, 1] += top_pad_expanded
            gt_hand_left_keypoints_2d[:, :, 0] += left_pad_expanded
            gt_hand_left_keypoints_2d[:, :, 1] += top_pad_expanded
            gt_hand_right_keypoints_2d[:, :, 0] += left_pad_expanded
            gt_hand_right_keypoints_2d[:, :, 1] += top_pad_expanded
            gt_face_keypoints_2d[:, :, 0] += left_pad_expanded
            gt_face_keypoints_2d[:, :, 1] += top_pad_expanded

            gt_face_keypoints_2d[:, :, :2] *= scales.view(-1, 1, 1)
            gt_hand_left_keypoints_2d[:, :, :2] *= scales.view(-1, 1, 1)
            gt_hand_right_keypoints_2d[:, :, :2] *= scales.view(-1, 1, 1)
            gt_pose_keypoints_2d[:, :, :2] *= scales.view(-1, 1, 1)

            # self.draw_kpts_on_image(cropped_images_hd, gt_pose_keypoints_2d, 
            #                         image_name_prefix, epoch, part = 'pose', stage = '2')

            ### get hand and face bboxes on hd_images
            hd_left_hand_bboxes = get_bboxes_from_keypoints(gt_hand_left_keypoints_2d, 1.15, (self.hd_size, self.hd_size))
            hd_right_hand_bboxes = get_bboxes_from_keypoints(gt_hand_right_keypoints_2d, 1.15, (self.hd_size, self.hd_size))
            hd_face_bboxes = get_bboxes_from_keypoints(gt_face_keypoints_2d, 1.2, (self.hd_size, self.hd_size))
            part_bboxes = {'left_hand': hd_left_hand_bboxes, 'right_hand': hd_right_hand_bboxes, 'head': hd_face_bboxes}

            ## crop hands, face image from cropped_images_hd
            # left_hand_image, _,_,_ = crop_resize_image_batch(cropped_images_hd, hd_left_hand_bboxes, self.input_size)
            # right_hand_image, _,_,_ = crop_resize_image_batch(cropped_images_hd, hd_right_hand_bboxes, self.input_size)
            # face_image, _,_,_ = crop_resize_image_batch(cropped_images_hd, hd_face_bboxes, self.input_size)
            # self.draw_kpts_on_image(left_hand_image, image_name_prefix, epoch, part = 'hand', stage = 'lh', )
            # self.draw_kpts_on_image(right_hand_image, image_name_prefix, epoch, part = 'hand', stage = 'rh')
            # self.draw_kpts_on_image(face_image, image_name_prefix, epoch, part = 'face', stage = 'fa')

            cropped_images = F.interpolate(cropped_images_hd, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
            scale_2 =  self.input_size/self.hd_size
            
            gt_pose_keypoints_2d[:, :, :2] *= scale_2
            gt_hand_left_keypoints_2d[:, :, :2] *= scale_2
            gt_hand_right_keypoints_2d[:, :, :2] *= scale_2
            gt_face_keypoints_2d[:, :, :2] *= scale_2
            
            new_batch = {'image':cropped_images,
                         'image_hd':cropped_images_hd,}


            data = {
                'body': new_batch
            }
            param_dict = self.model.encode(data, threthold=True, keep_local=True, copy_and_paste=False, part_bboxes = part_bboxes)
            
            # param_dict = pixie.encode(data, threthold=True, keep_local=True, copy_and_paste=True)
            # only use body params to get smplx output. TODO: we can also show the results of cropped head/hands
            codedict = param_dict['body']
            # for k in codedict:
            #     if codedict[k].shape[1] > 6: 
            #         print(k, codedict[k].shape)
            #     else:
            #         print(k, codedict[k].shape, codedict[k])

            prediction = self.model.decode(codedict, param_type='body')
            '''
            prediction = {
                    'vertices': verts,
                    'transformed_vertices': trans_verts,
                    'face_kpt': projected_landmarks,
                    'smplx_kpt': projected_joints,
                    'smplx_kpt3d': smplx_kpt3d,
                    'joints': joints,
                    'cam': cam,
                    }
            '''

            projected_2d_joints = prediction['smplx_kpt']
            pred_joints_3d = prediction['joints']

            pred_openpose_keypoints_2d = projected_2d_joints[:, self.openpose_in_smplx_mapping]
            pred_openpose_keypoints_3d = pred_joints_3d[:, self.openpose_in_smplx_mapping]
            pred_pose_keypoints_3d = pred_openpose_keypoints_3d[:, :25]


            gt_openpose_keypoints_2d = torch.cat([gt_pose_keypoints_2d, gt_hand_left_keypoints_2d, 
                                               gt_hand_right_keypoints_2d, gt_face_keypoints_2d], dim=1)
            
            gt_openpose_keypoints_2d_coord = gt_openpose_keypoints_2d[:, :, :2]
            gt_openpose_keypoints_2d_valid_mask = gt_openpose_keypoints_2d[:, :, 2] > points_confidence_threshold
            gt_openpose_keypoints_2d_confidence = gt_openpose_keypoints_2d[:, :, 2]

            start, end = 0, 0
            for part in self.part_count_dict:
                step = self.part_count_dict[part]
                start += end
                end += step

            loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            if self.compute_2d_loss:
                loss_2d = self.criteria(pred_openpose_keypoints_2d[:, start:end], gt_openpose_keypoints_2d_coord[:, start:end], 
                                    # confidence = gt_openpose_keypoints_2d_confidence[:, start:end],
                                     valid_mask = gt_openpose_keypoints_2d_valid_mask[:, start:end],
                                    )
                loss = loss + loss_2d
            

            if self.compute_3d_loss:
                ## gt_pose_keypoints_3d is the normalization of gt_pose_keypoints_2d by image size, give the 3d keypoints below
                ### this is just for debugging, we can also use the gt_pose_keypoints_2d directly if we have the 3d keypoints
                gt_pose_keypoints_3d = gt_joints_3d[:, self.openpose_in_smplx_mapping][:, :25]

                loss_3d = self.criteria(pred_pose_keypoints_3d, gt_pose_keypoints_3d, 
                                    # confidence = gt_openpose_keypoints_2d_confidence[:, start:end],
                                     valid_mask = gt_openpose_keypoints_2d_valid_mask[:, start:end],
                                    )
                loss = loss + loss_3d
            # print('loss:', loss.item())
            # backward & optimize
            loss.backward()
            self.optimizer.step()

            loginfo = f'{formatted_split} Epoch: {epoch:03d}/{self.max_epoch:03d}, Iter: {idx:05d}/{idx:05d}, Loss: {loss.item():.4f}'
            tbar.set_description(loginfo)
            loss_epoch.append(loss.item())
            if self.debug:
                break
        
        ## draw keypoints on image
        if self.dump_img and self.compute_2d_loss:                            
            self.draw_kpts_on_image(cropped_images, image_name_prefix, epoch, 
                                    gt_kpts_2d = gt_openpose_keypoints_2d, 
                                    pred_kpts_2d= pred_openpose_keypoints_2d, 
                                    part = 'pose', stage = '3')
        if self.compute_3d_loss:
            print('gt_pose_keypoints_3d:\n', gt_pose_keypoints_3d.shape, '\n', gt_pose_keypoints_3d[:, :4])         
            print('pred_pose_keypoints_3d\n:', pred_pose_keypoints_3d.shape, '\n', pred_pose_keypoints_3d[:, :4])         
            print('')
        loss_epoch = sum(loss_epoch)/len(loss_epoch)
        return loss_epoch
    
    def run(self):
        for epoch in range(self.max_epoch):
            loss_epoch = self.train_epoch(epoch)
            if loss_epoch < self.best_epoch_loss:
                best_epoch_loss = loss_epoch
                best_epoch = epoch
                self.logger.info(f"Best Epoch: {best_epoch}, Loss: {best_epoch_loss}")
                
                
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_epoch_loss': best_epoch_loss,
                }

                self.save_checkpoint(checkpoint_dict, self.log_dir, is_best = True)

            self.logger.info(f"Epoch: {epoch}, Loss: {loss_epoch}")
            self.scheduler.step(loss_epoch, epoch)
            if self.debug:
                break
            self.logger.info('')
        self.logger.info(f"Finish training")
    
    def draw_kpts_on_image(self, image, img_names, epoch, gt_kpts_2d = None, 
                           pred_kpts_2d = None, part = 'pose', stage = '3', gt_bboxes = None, pred_bboxes = None):
        '''
        image: (bs, H, W, 3)
        gt_kpts: (bs, N, 2)
        pred_kpts: (bs, N, 2)
        img_name: str list
        bboxes: (bs, 4)
        '''
        gt_color = (0, 255, 0)
        pred_color = (0, 0, 255)
        # print(f'\n stage: {stage}')
        # print('gt_kpts_2d:', gt_kpts_2d.shape) ##torch.Size([bs, 137, 3])
        
        for i in range(image.shape[0]):
            img_name = f'epoch{epoch:03d}_stage{stage}__{img_names[i]}'
            # img_name = f'epoch{epoch:03d}_{img_names[i]}'
            img = image[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = (img * 255).astype(np.uint8)
            img = img[:,:,::-1]
                        
            if gt_bboxes is not None or gt_kpts_2d is not None:
                img_for_gt = img.copy()
                if gt_bboxes is not None:
                    gt_bbox = gt_bboxes[i].cpu().numpy()
                    cv2.rectangle(img_for_gt, (int(gt_bbox[0]), int(gt_bbox[1])), 
                                            (int(gt_bbox[2]), int(gt_bbox[3]),), (0, 255, 0), 2)
            else:
                img_for_gt = None
                
            if pred_bboxes is not None or pred_kpts_2d is not None:
                img_for_pred = img.copy()
                if pred_bboxes is not None:
                    pred_bbox = pred_bboxes[i].cpu().numpy()
                    cv2.rectangle(img_for_pred, (int(pred_bbox[0]), int(pred_bbox[1])), 
                                            (int(pred_bbox[2]), int(pred_bbox[3]),), (0, 0, 255), 2)
            else:
                img_for_pred = None
                
            if part == 'pose':
                start = 0
                end = 25
            elif part == 'hand':
                start = 25
                end = 25 + 21 + 21
            elif part == 'face':
                start = 25 + 21 + 21
                end = 25 + 21 + 21 + 70

            if not gt_kpts_2d is None:
                gt_kpts = gt_kpts_2d[i].detach().cpu().numpy()
                gt_kpts = gt_kpts[start:end]
                # print('gt_kpts', gt_kpts.shape)
                # print('gt_kpts:', gt_kpts.shape, compute_mean_xy(gt_kpts), gt_kpts[:4])
                print('gt_kpts:', gt_kpts.shape, '\n', gt_kpts[:4,:2])
                
                ## compute mean x y of keypoints


                for kpt in gt_kpts:
                    x, y = kpt[:2]
                    cv2.circle(img_for_gt, (int(x), int(y)), 2, gt_color, -1)

            
            if pred_kpts_2d is not None:
                pred_kpts = pred_kpts_2d[i].detach().cpu().numpy()
                pred_kpts = pred_kpts[start:end]
                print('pred_kpts:', pred_kpts.shape, '\n', pred_kpts[:4])
                
                for kpt in pred_kpts:
                    x, y = kpt[:2]
                    cv2.circle(img_for_pred, (int(x), int(y)), 2, pred_color, -1)              
                
            if not img_for_gt is None and not img_for_pred is None:
                imgs = np.concatenate([img_for_gt, img_for_pred], axis = 1)
            elif not img_for_gt is None:
                imgs = img_for_gt
            elif not img_for_pred is None:
                imgs = img_for_pred
            else:
                imgs = img
            
            img_filepath = os.path.join(self.dump_img_dir, f'{img_name}.jpg')
            cv2.imwrite(img_filepath, imgs)

            break
            if self.debug:
                break

            if i > 3:
                break

    
    def save_checkpoint(self, dict_saved, save_dir, is_best = False):
        if is_best:
            torch.save(dict_saved, f"{save_dir}/best_checkpoint.pt")
        else:
            # torch.save(dict_saved, f"{save_dir}/checkpoint.pt")    
            pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PIXIE')
    parser.add_argument('--data_dir', type=str, default=None, help='path to dataset')
    parser.add_argument('--log_dir', type=str, default='log', help='path to save the log')
    parser.add_argument('--resume', type=str, default=None, help='path to the checkpoint')
    parser.add_argument('--fine_tune', action='store_true', help='fine tune the model')
    parser.add_argument('--useTex', default=True, action='store_true', help='use texture')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--max_epoch', type=int, default=100, help='max epoch')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--fast_train', action='store_true', help='fast train')
    parser.add_argument('--compute_2d_loss', action='store_true', help='compute loss in 2d')
    parser.add_argument('--compute_3d_loss', action='store_true', help='compute loss in 3d')

    args = parser.parse_args()

    worker = Worker(args = args)
    worker.run()