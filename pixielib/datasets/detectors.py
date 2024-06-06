import numpy as np
import torch
import torchvision.models.detection as models
import cv2, os, sys


'''
For cropping body:
1. using bbox from objection detectors
2. calculate bbox from body joints regressor

object detectors:
    know body object from label number
    https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    label for peopel: 1
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
'''
# TODO: add hand detector

#-- detetion
class FasterRCNN(object):
    ''' detect body
    '''
    def __init__(self, device='cuda:0'):  
        '''
        https://pytorch.org/docs/stable/torchvision/models.html#faster-r-cnn
        '''
        import torchvision
        # self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=models.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.input_size = 256
        print('FasterRCNN initialized.................................')

    @torch.no_grad()
    def run(self, input):
        '''
        input: 
            The input to the model is expected to be a list of tensors, 
            each of shape [C, H, W], one for each image, and should be in 0-1 range. 
            Different images can have different sizes.
        return: 
            detected box, [x1, y1, x2, y2]
        '''
        bs, c, h, w = input.size()
        if not h == self.input_size or not w == self.input_size:
            input = torch.nn.functional.interpolate(input, size=(self.input_size, self.input_size), mode='bilinear')
        scale_x = w/self.input_size
        scale_y = h/self.input_size
        prediction = self.model(input.to(self.device))[0]
        inds = (prediction['labels']==1)*(prediction['scores']>0.5)
        if len(inds) < 1:
            return None
        else:
            bbox = prediction['boxes'][inds][0].cpu().numpy()
            bbox[0] = int(bbox[0]*scale_x)
            bbox[1] = int(bbox[1]*scale_y)
            bbox[2] = int(bbox[2]*scale_x)
            bbox[3] = int(bbox[3]*scale_y)
            return bbox
    
    @torch.no_grad()
    def run_multi(self, input):
        '''
        input: 
            The input to the model is expected to be a list of tensors, 
            each of shape [C, H, W], one for each image, and should be in 0-1 range. 
            Different images can have different sizes.
        return: 
            detected box, [x1, y1, x2, y2]
        '''
        bs, c, h, w = input.size()
        if not h == self.input_size or not w == self.input_size:
            input = torch.nn.functional.interpolate(input, size=(self.input_size, self.input_size), mode='bilinear')
        scale_x = w/self.input_size
        scale_y = h/self.input_size
        prediction = self.model(input.to(self.device))[0]
        inds = (prediction['labels']==1)*(prediction['scores']>0.9)
        if len(inds) < 1:
            return None
        else:
            bbox = prediction['boxes'][inds].cpu().numpy()
            bbox[:,0] = bbox[:,0]*scale_x
            bbox[:,1] = bbox[:,1]*scale_y
            bbox[:,2] = bbox[:,2]*scale_x
            bbox[:,3] = bbox[:,3]*scale_y
            return bbox
    
    ### run a batch of images
    @torch.no_grad()
    def run_batch(self, input):
        '''
        input: 
            The input to the model is expected to be a list of tensors, 
            each of shape [bs, C, H, W], one for each image, and should be in 0-1 range. 
            Different images can have different sizes.
        return: 
            detected box, [x1, y1, x2, y2]
        '''
        bs, c, h, w = input.size()
        if not h == self.input_size or not w == self.input_size:
            input = torch.nn.functional.interpolate(input, size=(self.input_size, self.input_size), mode='bilinear')
        scale_x = w/self.input_size
        scale_y = h/self.input_size

        prediction = self.model(input.to(self.device))
        bbox_list = []
        for pred in prediction:
            # print('pred', pred)
            # 逻辑与操作来筛选标签为1且得分大于0.5的检测框
            inds = (pred['labels'] == 1) & (pred['scores'] > 0.6)
            if inds.sum() == 0:
                # 添加全零的边界框作为占位符
                # bbox = np.zeros(4)
                bbox = np.array([0, 0, w-1, h-1])
            else:
                # 使用非零索引获取第一个符合条件的框
                first_true_ind = inds.nonzero(as_tuple=True)[0][0]
                bbox = pred['boxes'][first_true_ind].cpu().numpy()
                bbox[0] = int(bbox[0]*scale_x)
                bbox[1] = int(bbox[1]*scale_y)
                bbox[2] = int(bbox[2]*scale_x)
                bbox[3] = int(bbox[3]*scale_y)

            bbox_list.append(bbox)
        # print('bbox_list', bbox_list)
        return np.array(bbox_list, dtype=np.int32)


# TODO
class Yolov4(object):
    def __init__(self, device='cuda:0'):
        print('Yolov4 initialized.....................................')

        pass

    @torch.no_grad()
    def run(self, image):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        pass

#-- person keypoints detection
# tested, not working well
class KeypointRCNN(object):
    ''' Constructs a Keypoint R-CNN model with a ResNet-50-FPN backbone.
    Ref: https://pytorch.org/docs/stable/torchvision/models.html#keypoint-r-cnn
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    '''
    def __init__(self, device='cuda:0'):  
        print('KeypointRCNN initialized.........................')

        import torchvision
        # self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights = models.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)

        self.model.to(device)
        self.model.eval()
        self.device = device
        self.input_size = 256

    @torch.no_grad()
    def run(self, input):
        '''
        input: 
            The input to the model is expected to be a list of tensors, 
            each of shape [C, H, W], one for each image, and should be in 0-1 range. 
            Different images can have different sizes.
        return: 
            boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values of x between 0 and W and values of y between 0 and H
            labels (Int64Tensor[N]): the class label for each ground-truth box
            keypoints (FloatTensor[N, K, 3]): the K keypoints location for each of the N instances, in the format [x, y, visibility], where visibility=0 means that the keypoint is not visible.
        '''
        prediction = self.model(input.to(self.device))[0]
        # 
        kpt = prediction['keypoints'][0].cpu().numpy()
        left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
        top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
        bbox = [left, top, right, bottom]
        return bbox, kpt
    
    @torch.no_grad()
    def run_batch(self, input):
        '''
        input: 
            The input to the model is expected to be a list of tensors, 
            each of shape [bs, C, H, W], one for each image, and should be in 0-1 range. 
            Different images can have different sizes.
        return: 
            detected box, [x1, y1, x2, y2]
        '''

        prediction = self.model(input.to(self.device))
        bbox_list = []
        kpt_list = []
        for pred in prediction:
            kpt = pred['keypoints'][0].cpu().numpy()
            left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
            top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
            bbox = [left, top, right, bottom]
            bbox_list.append(bbox)
            kpt_list.append(kpt)
        return np.array(bbox_list, dtype=np.int32), np.array(kpt_list, dtype=np.int32)

#-- face landmarks (68)
class FAN(object):
    def __init__(self):
        import face_alignment
        self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    def run(self, image):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        out = self.model.get_landmarks(image)
        if out is None:
            return [0]
        else:
            kpt = out[0].squeeze()
            left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
            top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
            bbox = [left,top, right, bottom]
            return bbox




if __name__ == "__main__":
    # testpath = r'C:\Users\hongr\Documents\GMU_research\computerVersion\hand_modeling\PIXIE\TestSamples\body\pexels-andrea-piacquadio-972937.jpg'
    testpath = r'C:\Users\hongr\Documents\GMU_research\computerVersion\hand_modeling\PIXIE\TestSamples\body\GettyImages-545880635.jpg'
    testpath = r'C:\Users\hongr\Documents\GMU_research\computerVersion\hand_modeling\PIXIE\TestSamples\body\4.png'
    img = cv2.imread(testpath)
    height, width, _ = img.shape
    if height > 800 or width > 800:
        length = max(height, width)
        scale = 800/length
        img = cv2.resize(img, (int(width*scale), int(height*scale)))

    imgs = [img, img]
    imgs = np.array(imgs)
    print('imgs', imgs.shape)

    imgs = torch.tensor(imgs.transpose(0,3,1,2), dtype=torch.float32)/255.
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    KeypointRCNN_detector = KeypointRCNN(device)
    FasterRCNN_detector = FasterRCNN(device)
    
    bbox_0 = FasterRCNN_detector.run_batch(imgs)
    # print('bbox_0', bbox_0.shape)
    bbox_1, kpt = KeypointRCNN_detector.run_batch(imgs)
    # print('bbox', bbox)
    print('kpt', kpt)

    # draw bbox and keypoints on image
    for i in range(len(imgs)):
        img = imgs[i].numpy().transpose(1,2,0)
        img = cv2.rectangle(img, (bbox_1[i][0], bbox_1[i][1]), (bbox_1[i][2], bbox_1[i][3]), (0,255,0), 2)
        img = cv2.rectangle(img, (bbox_0[i][0], bbox_0[i][1]), (bbox_0[i][2], bbox_0[i][3]), (0,125,255), 2)
        for j in range(kpt.shape[1]):
            img = cv2.circle(img, (kpt[i,j,0], kpt[i,j,1]), 2, (0,0,255), -1)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        break