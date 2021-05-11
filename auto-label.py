

import matplotlib.pyplot as plt
import os.path as osp 
import sscv 
import numpy as np 

from psd_tools import PSDImage
from pytoshop.user import nested_layers
from pytoshop import enums
from pytoshop.image_data import ImageData
from glob import glob 
from tqdm import tqdm

from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

# Input directory information 
# make auto-label dir as a argparse variable 

#TODO : make these inputs as argparse inputs 
#TODO : add erosion to damage area 

# auto_label_dir = "/home/sss/UOS-SSaS Dropbox/05. Data/06. Auto Labeling"

# dataset_name = 'General Crack'
# raw_img_dir_name = '/home/sss/UOS-SSaS Dropbox/05. Data/06. Auto Labeling/General Crack/01. Raw Image'
# rsz_img_dir_name = '02. Resized Image'
# pre_label_dir_name = '03. Pre-labeled Psd Files'
# corrected_label_dir_name = '04. Corrected Psd files' 
# rsz_img_size = 2048

# config_file = '/home/sss/UOS-SSaS Dropbox/05. Data/03. Checkpoints/2020.09.02_deeplabv3plus_r101-d8_769x769_40k_concrete_damage_cs/deeplabv3plus_r101-d8_769x769_40k_concrete_damage_cs.py'
# checkpoint_file = '/home/sss/UOS-SSaS Dropbox/05. Data/03. Checkpoints/2020.09.02_deeplabv3plus_r101-d8_769x769_40k_concrete_damage_cs/iter_40000.pth'


auto_label_dir = "/home/sss/UOS-SSaS Dropbox/05. Data/06. Auto Labeling"

dataset_name = 'KCQR'
raw_img_dir_name = '/home/sss/UOS-SSaS Dropbox/05. Data/06. Auto Labeling/KCQR/01. Raw Image'
rsz_img_dir_name = '02. Resized Image'
pre_label_dir_name = '03. Pre-labeled Psd Files'
corrected_label_dir_name = '04. Corrected Psd files' 
rsz_img_size = 1024

config_file = '/home/sss/UOS-SSaS Dropbox/05. Data/03. Checkpoints/2020.09.02_deeplabv3plus_r101-d8_769x769_40k_concrete_damage_cs/deeplabv3plus_r101-d8_769x769_40k_concrete_damage_cs.py'
checkpoint_file = '/home/sss/UOS-SSaS Dropbox/05. Data/03. Checkpoints/2020.09.02_deeplabv3plus_r101-d8_769x769_40k_concrete_damage_cs/iter_40000.pth'

num_of_class = 4 # not include background 

palette = [
    [255, 0, 0], # crack 
    [0, 255, 0], # effl 
    [0, 255, 255], # rebar
    [255, 255, 0], # spalling
]

classes = ['crack', 'efflorescence', 'rebar', 'spalling']


def create_layer_from_detection_result(detection_result, class_num, class_name, color_map):

    """
    Args :
        detection_result (list) : detection result of mmsegmentation model 
        class_num (int or float) : class number 
        class_name (str) : class_name 
        color_map (list) : RGB value of color to highlight damage area

    Return : 
        damage_layer (pytoshop nested layer) : layer with damage detection 

    """

    height, width = detection_result[0].shape[0], detection_result[0].shape[1]
     
    # create white RGBA image with white color 
    detection_result_with_transparent_bg = np.ones((height, width, 4) , dtype = np.uint8)*255


    damage_area = detection_result[0] == class_num
    non_damage_area = detection_result[0] != class_num
    detection_result_with_transparent_bg[damage_area, 0] =  color_map[0]
    detection_result_with_transparent_bg[damage_area, 1] =  color_map[1]
    detection_result_with_transparent_bg[damage_area , 2] =  color_map[2]
    detection_result_with_transparent_bg[non_damage_area , 3] =  0

    # create image channel dictionary according to pytoshop nested_layers' input
    lyr_channels = {}

    lyr_channels[-1] = detection_result_with_transparent_bg[:,:,3]
    lyr_channels[0] = detection_result_with_transparent_bg[:,:,0]
    lyr_channels[1] = detection_result_with_transparent_bg[:,:,1]
    lyr_channels[2] = detection_result_with_transparent_bg[:,:,2]

    damage_layer = nested_layers.Image(name=class_name,
                                visible=True, opacity=255, group_id=0,
                                blend_mode=enums.BlendMode.normal, 
                                top=0, left=0, 
                                bottom=height, right=width, channels=lyr_channels,
                                metadata=None, layer_color=0, color_mode=None)

    return damage_layer

def pre_label_imgs(rsz_img_dir, pre_label_dir):

    """
    Args : 
        rsz_img_dir   : directory of resized image files 
        pre_label_dir : directory of pre-labeled image files 

    Return : 
        None : This function directly saves images in pre_label_dir

    """

    # Find difference btw raw and resized img lists 

    pre_label_list = [osp.basename(x) for x in glob(osp.join(pre_label_dir, '*.psd'))]
    print("{} images read from the psd files folder.".format(len(pre_label_list)))

    rsz_img_list = [osp.basename(x) for x in glob(osp.join(rsz_img_dir, '*.jpg'))]
    print("{} images read from the resized image folder.".format(len(rsz_img_list)))


    print("Compare resized image list and psd file list....")
    imgs_to_pre_label_list = [img_path for img_path in rsz_img_list if img_path not in pre_label_list]
    print("{} images are found not in the psd files folder.".format(len(imgs_to_pre_label_list)))

    # build the model from a config file and a checkpoint file
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

    for img_filename in tqdm(imgs_to_pre_label_list): 

        img_basename = osp.basename(img_filename).split('.')[0]
        img_filepath = osp.join(rsz_img_dir, img_basename + '.jpg')
        
        img = sscv.imread(img_filepath)

        psd_layers = []

        bg_channels = [img[:,:,0], img[:,:,1], img[:,:,2]]
                
        bg_height, bg_width = img.shape[0], img.shape[1] 

        bg_layer = nested_layers.Image(name='Background',
                                    visible=True, opacity=255, group_id=0,
                                    blend_mode=enums.BlendMode.normal, 
                                    top=0, left=0, 
                                    bottom=bg_height, right=bg_width, channels=bg_channels,
                                    metadata=None, layer_color=0, color_mode=None)

        detection_result = inference_segmentor(model, img_filepath)

        for class_num in range(num_of_class):
            
            class_name = classes[class_num]
            color_map = palette[class_num]
            damage_layer = create_layer_from_detection_result(detection_result, class_num+1, class_name, color_map) # add 1 to class num to avoid background
            psd_layers.append(damage_layer)
            

        psd_layers.append(bg_layer)

        output = nested_layers.nested_layers_to_psd(psd_layers, color_mode=3, depth=8,
                                                    size=(bg_width, bg_height))

        save_dir = osp.join(pre_label_dir, img_basename + '.psd')

        with open(save_dir, 'wb') as fd:
            output.write(fd)


def resize_and_move_imgs(raw_img_dir, rsz_img_dir): 

    """
    Args : 
        raw_img_dir : directory of raw image files 
        rsz_img_dir : directory of resized image files 

    Return : 
        None : This function directly saves images in rsz_img_dir

    """

    # Find difference btw raw and resized img lists 
    print("Step 1. Raw Image Folder to Resized Image Folder ... ")
    print("    ... Start resize and copy images from raw image folder to resized image folder...")
    raw_img_list = [osp.basename(x) for x in glob(osp.join(raw_img_dir, '*.jpg'))]
    print("{} images read from the raw image folder.".format(len(raw_img_list)))

    rsz_img_list = [osp.basename(x) for x in glob(osp.join(rsz_img_dir, '*.jpg'))]
    print("{} images read from the resized image folder.".format(len(rsz_img_list)))

    print("Compare raw image list and resized image list....")
    imgs_to_rsz_list = [img_path for img_path in raw_img_list if img_path not in rsz_img_list]
    print("{} images are found not in the resized image folder.".format(len(imgs_to_rsz_list)))


    # Move image from raw folder to resize folder 
    for idx, img_filename in enumerate(imgs_to_rsz_list): 
        org_img_path = osp.join(raw_img_dir, img_filename)
        img = sscv.imread(org_img_path)
        rsz_img_path = osp.join(rsz_img_dir, img_filename)
        
        is_height_longer = img.shape[0] > img.shape[1]
        
        if is_height_longer : 
            rsz_img = sscv.resize(img, height = rsz_img_size)
        else : 
            rsz_img = sscv.resize(img, width = rsz_img_size)
            
        print("Image Num: {}, Original Image Shape: {}, Resized Image Shape : {}, Image Filename: {} "
            .format(idx, img.shape, rsz_img.shape, img_filename)
            )
        
        sscv.imwrite(rsz_img_path, rsz_img)


def main(): 

    print("Start Auto Labeling Procedure ....")

    dataset_dir = osp.join(auto_label_dir, dataset_name)
    raw_img_dir = osp.join(raw_img_dir_name)
    rsz_img_dir = osp.join(dataset_dir, rsz_img_dir_name)
    pre_label_dir = osp.join(dataset_dir, pre_label_dir_name)
    corrected_label_dir = osp.join(dataset_dir, corrected_label_dir_name) 

    resize_and_move_imgs(raw_img_dir, rsz_img_dir)

    pre_label_imgs(rsz_img_dir, pre_label_dir)


if __name__ == "__main__" : 
    main()
    
