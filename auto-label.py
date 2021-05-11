
import argparse
import sscv 
import cv2

import os.path as osp 
import numpy as np 

from pytoshop.user import nested_layers
from pytoshop import enums
from pytoshop.image_data import ImageData
from glob import glob 
from tqdm import tqdm

from mmseg.apis import init_segmentor, inference_segmentor

# Input directory information 
# make auto-label dir as a argparse variable 

#TODO : reduce required arguments 



def create_layer_from_detection_result(detection_result, class_num, class_name, color_map,
                                        erode_size = 0):

    """
    Args :
        detection_result (list) : detection result of mmsegmentation model 
        class_num (int or float) : class number 
        class_name (str) : class_name 
        color_map (list) : RGB value of color to highlight damage area
        erode_size (int)

    Return : 
        damage_layer (pytoshop nested layer) : layer with damage detection 

    """

    height, width = detection_result[0].shape[0], detection_result[0].shape[1]

    damage_area = detection_result[0] == class_num

    if not erode_size == 0 : 
        damage_area = np.asarray(damage_area, dtype = np.uint8)
        kernel = np.ones((erode_size,erode_size),np.uint8)
        damage_area = cv2.erode(damage_area, kernel, iterations = 1)
        damage_area = np.asarray(damage_area, dtype = bool)

    non_damage_area = damage_area == 0

    # create white RGBA image with white color 
    detection_result_with_transparent_bg = np.ones((height, width, 4) , dtype = np.uint8)*255
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

def pre_label_imgs(rsz_img_dir, pre_label_dir, model_config_file, 
                    model_checkpoint_file, num_of_class, palette, 
                    classes, erode_size):

    """
    Args : 
        rsz_img_dir   : directory of resized image files 
        pre_label_dir : directory of pre-labeled image files 
        model_config_file : 
        model_checkpoint_file :
        num_of_class
        palette
        classes
        erode_size 

    Return : 
        None : This function directly saves images in pre_label_dir

    """

    # Find difference btw raw and resized img lists 

    pre_label_list = [osp.basename(x) for x in glob(osp.join(pre_label_dir, '*.psd'))]
    print("{} images read from the psd files folder.".format(len(pre_label_list)))

    rsz_img_list = [osp.basename(x) for x in glob(osp.join(rsz_img_dir, '*.jpg'))]
    print("{} images read from the resized image folder.".format(len(rsz_img_list)))

    print("Compare resized image list and psd file list....")
    imgs_to_pre_label_list = [img_path for img_path in rsz_img_list if img_path[:-4] + '.jpg' not in pre_label_list]
    print("{} images are found not in the psd files folder.".format(len(imgs_to_pre_label_list)))

    # build the model from a config file and a checkpoint file
    model = init_segmentor(model_config_file, model_checkpoint_file, device='cuda:0')

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
            damage_layer = create_layer_from_detection_result(detection_result, class_num+1, 
                                                                class_name, color_map, erode_size=erode_size) # add 1 to class num to avoid background
            psd_layers.append(damage_layer)
            

        psd_layers.append(bg_layer)

        output = nested_layers.nested_layers_to_psd(psd_layers, color_mode=3, depth=8,
                                                    size=(bg_width, bg_height))

        save_dir = osp.join(pre_label_dir, img_basename + '.psd')

        with open(save_dir, 'wb') as fd:
            output.write(fd)


def resize_and_move_imgs(raw_img_dir, rsz_img_dir, rsz_img_size): 

    """
    Args : 
        raw_img_dir : directory of raw image files 
        rsz_img_dir : directory of resized image files 
        rsz_img_size 

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


def main(auto_label_dir, dataset_name, rsz_img_size, model_config_file,
         model_checkpoint_file, num_of_class, palette, classes, erode_size): 

    print("Start Auto Labeling Procedure ....")

    dataset_dir = osp.join(auto_label_dir, dataset_name)
    raw_img_dir = osp.join(dataset_dir, '01. Raw Image')
    rsz_img_dir = osp.join(dataset_dir, '02. Resized Image')
    pre_label_dir = osp.join(dataset_dir, '03. Pre-labeled Psd Files')
    corrected_label_dir = osp.join(dataset_dir, '04. Corrected Psd files' ) 

    resize_and_move_imgs(raw_img_dir, rsz_img_dir, rsz_img_size)

    pre_label_imgs(rsz_img_dir, pre_label_dir,
                    model_config_file, model_checkpoint_file,
                    num_of_class, palette, classes, erode_size)


if __name__ == "__main__" : 
    
    parser = argparse.ArgumentParser(description='Get configuration file to load.')
    parser.add_argument('--config', type=str, help='configuration file path')
    args = parser.parse_args()
    config_filename = args.config

    if config_filename == 'kcqr_config.py' : 
        from kcqr_config import configs
    
    auto_label_dir = configs['auto_label_dir'] 
    dataset_name = configs['dataset_name'] 
    rsz_img_size = configs['rsz_img_size'] 
    model_config_file = configs['model_config_file'] 
    model_checkpoint_file = configs['model_checkpoint_file'] 
    num_of_class = configs['num_of_class'] 
    erode_size = configs['erode_size'] 
    palette = configs['palette'] 
    classes = configs['classes'] 
    
        
    main(auto_label_dir, dataset_name, rsz_img_size,
         model_config_file, model_checkpoint_file, 
         num_of_class, palette, classes, erode_size)
    
