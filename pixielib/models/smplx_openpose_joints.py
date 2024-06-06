import numpy as np

SMPLX_names = [  
                ### 25 body keypoints
                    'pelvis',      'left_hip',         'right_hip',        'spine1',       'left_knee', 
                    'right_knee',  'spine2',           'left_ankle',       'right_ankle',  'spine3', 
                    'left_foot',   'right_foot',       'neck',             'left_collar',  'right_collar',
                    'head',        'left_shoulder',    'right_shoulder',   'left_elbow',   'right_elbow', 
                    'left_wrist',  'right_wrist',      'jaw',              'left_eye_smplx', 'right_eye_smplx', 

                ### hand keypoints
                    'left_index1', 'left_index2', 'left_index3', 'left_middle1', 'left_middle2', 
                    'left_middle3', 'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_ring1', 
                    'left_ring2', 'left_ring3', 'left_thumb1', 'left_thumb2', 'left_thumb3', 

                    'right_index1', 'right_index2', 'right_index3', 'right_middle1', 'right_middle2', 
                    'right_middle3', 'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_ring1', 
                    'right_ring2', 'right_ring3', 'right_thumb1', 'right_thumb2', 'right_thumb3', 

                ### 51 facial landmarks
                    'right_eye_brow1', 'right_eye_brow2', 'right_eye_brow3', 'right_eye_brow4', 'right_eye_brow5', 
                    'left_eye_brow5',  'left_eye_brow4',  'left_eye_brow3',  'left_eye_brow2',  'left_eye_brow1', 
                    'nose1',           'nose2',           'nose3',           'nose4',           'right_nose_2', 
                    'right_nose_1',    'nose_middle',     'left_nose_1',     'left_nose_2',     'right_eye1', 
                    'right_eye2',      'right_eye3',      'right_eye4',      'right_eye5',      'right_eye6', 
                    'left_eye4',       'left_eye3',       'left_eye2',       'left_eye1',       'left_eye6', 
                    'left_eye5',       'right_mouth_1',   'right_mouth_2',   'right_mouth_3',   'mouth_top', 
                    'left_mouth_3',    'left_mouth_2',    'left_mouth_1',    'left_mouth_5',    'left_mouth_4', 
                    'mouth_bottom',    'right_mouth_4',   'right_mouth_5',   'right_lip_1',     'right_lip_2', 
                    'lip_top',         'left_lip_2',       'left_lip_1',     'left_lip_3',      'lip_bottom', 'right_lip_3', 
                    
                    ### 17 contour landmarks
                'right_contour_1', 'right_contour_2', 'right_contour_3', 'right_contour_4', 
                'right_contour_5', 'right_contour_6', 'right_contour_7', 'right_contour_8', 
                'contour_middle', 
                'left_contour_8', 'left_contour_7', 'left_contour_6', 'left_contour_5', 
                'left_contour_4', 'left_contour_3', 'left_contour_2', 'left_contour_1', 

                'head_top', 'left_big_toe', 'left_ear', 'left_eye', 'left_heel', 
                'left_index', 'left_middle', 'left_pinky', 'left_ring', 
                'left_small_toe', 'left_thumb', 'nose', 'right_big_toe', 
                'right_ear', 'right_eye', 'right_heel', 
                'right_index', 'right_middle', 'right_pinky', 'right_ring', 'right_small_toe', 'right_thumb'
                ]

# print('SMPLX_names', len(SMPLX_names)) ###145
facial_selected_keypoints = [
"right_eye_brow1", "right_eye_brow2", "right_eye_brow3", "right_eye_brow4", "right_eye_brow5",
"left_eye_brow1", "left_eye_brow2", "left_eye_brow3", "left_eye_brow4", "left_eye_brow5",
"right_eye1", "right_eye2", "right_eye3", "right_eye4", "right_eye5", "right_eye6",
"left_eye1", "left_eye2", "left_eye3", "left_eye4", "left_eye5", "left_eye6",
"mouth_top", "mouth_bottom",
"right_mouth_1", "right_mouth_2", "right_mouth_3", "right_mouth_4", "right_mouth_5",
"left_mouth_1", "left_mouth_2", "left_mouth_3", "left_mouth_4", "left_mouth_5",
# 'right_contour_1', 'left_contour_1',
]
main_body_selected_keypoints = [
"spine1",
"spine2",
"spine3",
"neck",
"left_collar",
"right_collar",
"left_shoulder",
"right_shoulder",
"left_elbow",
"right_elbow",
"left_wrist",
"right_wrist",
"jaw",
]
hands_selected_keypoints = [
"left_thumb1", "left_thumb2", "left_thumb3", "left_thumb", ## left_thumb is the tip
"left_index1", "left_index2", "left_index3", "left_index", ## left_index is the tip
"left_middle1", "left_middle2", "left_middle3", "left_middle", ## left_middle is the tip
"left_ring1", "left_ring2", "left_ring3", "left_ring",
"left_pinky1", "left_pinky2", "left_pinky3", "left_pinky",
"right_thumb1", "right_thumb2", "right_thumb3", "right_thumb",
"right_index1", "right_index2", "right_index3", "right_index",
"right_middle1", "right_middle2", "right_middle3", "right_middle",
"right_ring1", "right_ring2", "right_ring3", "right_ring",
"right_pinky1", "right_pinky2", "right_pinky3", "right_pinky",
]

selected_keypoints = main_body_selected_keypoints + hands_selected_keypoints + facial_selected_keypoints
selected_indicecs = [SMPLX_names.index(k) for k in selected_keypoints]
selected_mapping = np.array(selected_indicecs, dtype=np.int32)


selected_print_face_keypoints = [
        'right_eye_brow1', 'right_eye_brow2', 'right_eye_brow3', 'right_eye_brow4', 'right_eye_brow5', 
                    'left_eye_brow5',  'left_eye_brow4',  'left_eye_brow3',  'left_eye_brow2',  'left_eye_brow1', 
                    'nose1',           'nose2',           'nose3',           'nose4',           'right_nose_2', 
                    'right_nose_1',    'nose_middle',     'left_nose_1',     'left_nose_2',     'right_eye1', 
                    'right_eye2',      'right_eye3',      'right_eye4',      'right_eye5',      'right_eye6', 
                    'left_eye4',       'left_eye3',       'left_eye2',       'left_eye1',       'left_eye6', 
                    'left_eye5',       'right_mouth_1',   'right_mouth_2',   'right_mouth_3',   'mouth_top', 
                    'left_mouth_3',    'left_mouth_2',    'left_mouth_1',    'left_mouth_5',    'left_mouth_4', 
                    'mouth_bottom',    'right_mouth_4',   'right_mouth_5',   'right_lip_1',     'right_lip_2', 
                    'lip_top',         'left_lip_2',       'left_lip_1',     'left_lip_3',      'lip_bottom', 'right_lip_3', 
                    ]

print_keypoints = ['left_thumb', 'left_index', 'left_middle', 'left_pinky', 'right_contour_1', 'left_contour_1']
# print_indices = [SMPLX_names.index(k) for k in print_keypoints]
print_indices = [SMPLX_names.index(k) for k in selected_print_face_keypoints]
print_indices = np.array(print_indices, dtype=np.int32)

## write SMPLX_names to a txt file, every element in a line
# with open('SMPLX_names.txt', 'w') as f:
#     for item in SMPLX_names:
#         f.write("%s\n" % item)
# return
# return
openpose_body25_names = [
"Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "MidHip",
"RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar", "LBigToe",
"LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel", #"Background"
]

## convert openpose_body25_names with the names in SMPLX_names
openpose_body25_names_in_smplx = [
"nose", "neck", "right_shoulder", "right_elbow", "right_wrist", 
"left_shoulder", "left_elbow", "left_wrist", "pelvis",
"right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle", 
"right_eye", "left_eye", "right_ear", "left_ear", "left_big_toe",  # 20
"left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel", #"background"
]

openpose_left_hand_names = [
"LWrist", 
"LThumb1", "LThumb2", "LThumb3", "LThumb4", 
"LIndex1", "LIndex2", "LIndex3", "LIndex4", 
"LMiddle1", "LMiddle2", "LMiddle3", "LMiddle4", 
"LRing1", "LRing2", "LRing3", "LRing4", 
"LPinky1", "LPinky2", "LPinky3", "LPinky4",
]
## convert openpose_left_hand_names with the names in SMPLX_names
openpose_left_hand_names_in_smplx = [
'left_wrist',
'left_thumb1', 'left_thumb2', 'left_thumb3', 'left_thumb',
'left_index1', 'left_index2', 'left_index3', 'left_index',
'left_middle1', 'left_middle2', 'left_middle3', 'left_middle',
'left_ring1', 'left_ring2', 'left_ring3', 'left_ring',
'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_pinky',
]


openpose_right_hand_names = [
"RWrist",
"RThumb1", "RThumb2", "RThumb3", "RThumb4",
"RIndex1", "RIndex2", "RIndex3", "RIndex4",
"RMiddle1", "RMiddle2", "RMiddle3", "RMiddle4",
"RRing1", "RRing2", "RRing3", "RRing4",
"RPinky1", "RPinky2", "RPinky3", "RPinky4",
]

## convert openpose_right_hand_names with the names in SMPLX_names
openpose_right_hand_names_in_smplx = [
'right_wrist',
'right_thumb1', 'right_thumb2', 'right_thumb3', 'right_thumb',
'right_index1', 'right_index2', 'right_index3', 'right_index',
'right_middle1', 'right_middle2', 'right_middle3', 'right_middle',
'right_ring1', 'right_ring2', 'right_ring3', 'right_ring',
'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_pinky',
]


## face keypoints contains 70 points, including 51 facial landmarks and 17 contour landmarks
openpose_face_names_in_smplx = [
    # 0~16: face contour
    'right_contour_1', 'right_contour_2', 'right_contour_3', 'right_contour_4', 
    'right_contour_5', 'right_contour_6', 'right_contour_7', 'right_contour_8', 
    'contour_middle', 
    'left_contour_8', 'left_contour_7', 'left_contour_6', 'left_contour_5', 
    'left_contour_4', 'left_contour_3', 'left_contour_2', 'left_contour_1',

    # 17~21: right eyebrow
    'right_eye_brow1', 'right_eye_brow2', 'right_eye_brow3', 'right_eye_brow4', 'right_eye_brow5',
    # 22~26: left eyebrow
    'left_eye_brow5', 'left_eye_brow4', 'left_eye_brow3', 'left_eye_brow2', 'left_eye_brow1',

    # 27~30: nose
    'nose1', 'nose2', 'nose3', 'nose4',

    # 31~35: 
    'right_nose_2', 'right_nose_1', 'nose_middle', 'left_nose_1', 'left_nose_2',

    # 36~41: right eye
    'right_eye1', 'right_eye2', 'right_eye3', 'right_eye4', 'right_eye5', 'right_eye6',

    # 42~47: left eye
    'left_eye4', 'left_eye3', 'left_eye2', 'left_eye1', 'left_eye6', 'left_eye5',

    #48~59: mouth
    'right_mouth_1', 'right_mouth_2', 'right_mouth_3', 
    'mouth_top', 
    'left_mouth_3', 'left_mouth_2', 'left_mouth_1', 'right_mouth_5', 'right_mouth_4',
    'mouth_bottom',
    'right_mouth_4', 'right_mouth_5', 

    # 60~67: lip
    'right_lip_1', 'right_lip_2', 'lip_top', 'left_lip_2', 'left_lip_1', 'left_lip_3', 'lip_bottom', 'right_lip_3',

    # 68~69: 
    'right_eye', 'left_eye',   
]

openpose_kpts_names = openpose_body25_names_in_smplx + openpose_left_hand_names_in_smplx \
      + openpose_right_hand_names_in_smplx + openpose_face_names_in_smplx

openpose_in_smplx_indices = [SMPLX_names.index(k) for k in openpose_kpts_names]
openpose_in_smplx_mapping = np.array(openpose_in_smplx_indices, dtype=np.int32)
