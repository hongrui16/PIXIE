import os, sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import scipy
from PIL import Image

def crop_resize_image(image, bbox, scale, target_size):
    """
    根据边界框（bbox），缩放因子（scale）和目标尺寸（target_size）裁剪并调整图像尺寸。
    参数:
    - image: PIL.Image对象，原始图像。
    - bbox: 边界框，格式为(left, top, right, bottom)。
    - scale: 缩放因子，用于确定裁剪的正方形边长。
    - target_size: 要调整到的新尺寸（宽度和高度相同）。
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        raise ValueError('image must be a PIL.Image or numpy.ndarray object')
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top
    
    # 计算边界框的中心
    center_x = left + width / 2.0
    center_y = top + height / 2.0
    
    # 找出最长边并计算新的裁剪边长
    long_edge = max(width, height) * scale
    half_edge = long_edge / 2.0
    
    # 计算新的边界框
    new_left = int(center_x - half_edge)
    new_top = int(center_y - half_edge)
    new_right = int(center_x + half_edge)
    new_bottom = int(center_y + half_edge)
    
    # 裁剪图像
    cropped_image = image.crop((new_left, new_top, new_right, new_bottom))
    
    # 调整图像尺寸
    resized_image = cropped_image.resize((target_size, target_size), Image.ANTIALIAS)
    
    return np.array(resized_image)



def crop_resize_image_batch(image, bbox, target_size):
    """
    根据边界框（bbox），缩放因子（scale）和目标尺寸（target_size）裁剪并调整图像尺寸。
    参数:
    - image: torch.Tensor对象，原始图像，形状为(B, C, H, W)。
    - bbox: 边界框，形状为(B, 4)，格式为(left, top, right, bottom)。
    - target_size: 要调整到的新尺寸（宽度和高度相同）。
    返回:
    - cropped_images: 裁剪并调整大小后的图像，形状为(B, C, target_size, target_size)。
    - scales_x: 每个bbox的x缩放因子。
    - scales_y: 每个bbox的y缩放因子。
    - left_pad: 每个bbox的左填充。
    - top_pad: 每个bbox的上填充。
    """
    scale = 1 ## cannot be changed without changing the corresponding key points coordinates

    bs, c, h, w = image.shape
    
    # 避免修改原始bbox, convert to float
    new_bbox = bbox
    

    # # 缩放bbox
    # bbox_width = new_bbox[:, 2] - new_bbox[:, 0]
    # bbox_height = new_bbox[:, 3] - new_bbox[:, 1]
    # new_bbox[:, 0] -= (scale - 1) * bbox_width / 2
    # new_bbox[:, 2] += (scale - 1) * bbox_width / 2
    # new_bbox[:, 1] -= (scale - 1) * bbox_height / 2
    # new_bbox[:, 3] += (scale - 1) * bbox_height / 2
    
    # 确保bbox不越界
    # new_bbox = torch.clamp(new_bbox, 0, max(w-1, h-1))

    # 计算新的长边
    bbox_width = new_bbox[:, 2] - new_bbox[:, 0]
    bbox_height = new_bbox[:, 3] - new_bbox[:, 1]
    max_side = torch.max(bbox_width, bbox_height)

    # 为每个bbox计算填充
    left_pad = (max_side - bbox_width) / 2
    right_pad = max_side - bbox_width - left_pad
    top_pad = (max_side - bbox_height) / 2
    bottom_pad = max_side - bbox_height - top_pad
    
    # 初始化输出张量
    cropped_images = []
    # scales_x = target_size / bbox_width
    # scales_y = target_size / bbox_height

    scales = target_size / max_side

    for i in range(bs):
        # 裁剪图片
        img_cropped = image[i:i+1, :, int(new_bbox[i, 1]):int(new_bbox[i, 3]), int(new_bbox[i, 0]):int(new_bbox[i, 2])]
        # 填充图片
        img_padded = F.pad(img_cropped, (int(left_pad[i]), int(right_pad[i]), int(top_pad[i]), int(bottom_pad[i])))
        
        # 调整大小到目标尺寸
        img_resized = F.interpolate(img_padded, size=(target_size, target_size), mode='bilinear', align_corners=False)
        cropped_images.append(img_resized)
    cropped_images = torch.cat(cropped_images, dim=0)
    return cropped_images, scales, left_pad, top_pad


def pad_resize_image(frame, hd_size = 720):
    
    # find the longer side of the frame and then pad the shorter side to make it square, then resize to hd_size
    h, w, _ = frame.shape
    if h > w:
        pad_left = (h - w) // 2
        pad_right = h - w - pad_left
        frame = cv2.copyMakeBorder(frame, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        start_x = pad_left
        start_y = 0
        max_len = h
    else:
        pad_up = (w - h) // 2
        pad_down = w - h - pad_up
        frame = cv2.copyMakeBorder(frame, pad_up, pad_down, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        start_x = 0
        start_y = pad_up
        max_len = w

    frame = cv2.resize(frame, (hd_size, hd_size))
    scale = hd_size / max_len  # scale factor for the image
    

    return frame, scale, start_x, start_y

def get_bboxes_from_keypoints(keypoints, scale, img_size):
    '''
    inputs:
        keypoints: tensor, shape (B, N, 3)
        scale: float, scale factor for the image
        img_size: tuple, (h, w)    
    returns:
        bboxes: tensor, shape (B, 4), (left, top, right, bottom)    
    '''
    h,w = img_size
    bboxes = torch.zeros((keypoints.shape[0], 4), device=keypoints.device)

    ##find the min and max x and y coordinates
    min_coords, _ = torch.min(keypoints[:,:,:2], dim=1)
    xmin, ymin = min_coords[:, 0], min_coords[:, 1]
    max_coords, _ = torch.max(keypoints[:,:,:2], dim=1)
    xmax, ymax = max_coords[:, 0], max_coords[:, 1]

    ## concatenate the min and max coordinates to get the bounding box
    bboxes[:, 0] = xmin
    bboxes[:, 1] = ymin
    bboxes[:, 2] = xmax
    bboxes[:, 3] = ymax

    ## scale the bboxes, 
    bbox_width = bboxes[:, 2] - bboxes[:, 0]
    bbox_height = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] -= (scale - 1) * bbox_width / 2
    bboxes[:, 2] += (scale - 1) * bbox_width / 2
    bboxes[:, 1] -= (scale - 1) * bbox_height / 2
    bboxes[:, 3] += (scale - 1) * bbox_height / 2

    bboxes = torch.clamp(bboxes, 0, max(w-1, h-1))
    return bboxes

def compute_mean_xy(keypoints):
    """
    Compute mean x and y coordinates for the given keypoints.
    Keypoints are of shape [n, 3], where the 3 stands for x, y, confidence.
    """
    # Extract x and y coordinates where confidence is positive
    valid_keypoints = keypoints[:, :2] * (keypoints[:, 2:] > 0)
    
    # Compute mean x and y coordinates
    sum_x = valid_keypoints[:, 0].sum()
    sum_y = valid_keypoints[:, 1].sum()
    count = (keypoints[:, 2] > 0).sum()
    
    mean_x = sum_x / count
    mean_y = sum_y / count
    
    return mean_x, mean_y


if __name__ == '__main__':
    # 假设图片和边界框数据
    image = torch.rand(3, 300, 300)  # 300x300的图片
    bbox = torch.tensor([50, 50, 250, 250], dtype=torch.float)
    images = torch.rand(5, 3, 300, 300)  # 5张300x300的图片
    bboxes = torch.tensor([[50, 50, 250, 250], [30, 30, 200, 200], [60, 60, 240, 240], [20, 20, 220, 220], [40, 40, 230, 230]], dtype=torch.float)
    
    target_size = 224

    # 调用函数
    # output_images = crop_resize_image(image, bbox, scale, target_size)
    # print(output_images.shape)  # 应输出(5, 224, 224, 3)
    output_images, scales_x, scales_y, left_pad, top_pad = crop_resize_image_batch(images, bboxes, target_size)
    print(output_images.shape)  # 应输出(5, 3, 224, 224)
