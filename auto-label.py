

import matplotlib.pyplot as plt
import os.path as osp 
import sscv 

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
auto_label_dir = "/run/user/1000/gvfs/afp-volume:host=DiskStation.local,user=shm_nas,volume=Volume_2/06. Auto-Labeling"

dataset_name = 'General Crack'
raw_img_dir_name = '/home/sss/UOS-SSaS Dropbox/05. Data/06. Auto Labeling/General Crack/01. Raw Image'
rsz_img_dir_name = '02. Resized Image'
pre_label_dir_name = '03. Pre-labeled Psd Files'
corrected_label_dir_name = '04. Corrected Psd files' 
rsz_img_size = 2048

config_file = '/home/sss/UOS-SSaS Dropbox/05. Data/03. Checkpoints/2020.09.02_deeplabv3plus_r101-d8_769x769_40k_concrete_damage_cs/deeplabv3plus_r101-d8_769x769_40k_concrete_damage_cs.py'
checkpoint_file = '/home/sss/UOS-SSaS Dropbox/05. Data/03. Checkpoints/2020.09.02_deeplabv3plus_r101-d8_769x769_40k_concrete_damage_cs/iter_40000.pth'

dummy_psd_filepath = '/home/sss/UOS-SSaS Dropbox/05. Data/06. Auto Labeling/General Crack/01. Raw Image/dummy.psd'


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
    imgs_to_pre_label_list = [img_path for img_path in pre_label_list if img_path not in rsz_img_list]
    print("{} images are found not in the psd files folder.".format(len(imgs_to_pre_label_list)))

    # build the model from a config file and a checkpoint file
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

    for idx, img_filename in enumerate(imgs_to_pre_label_list): 

        img_filepath = osp.join(rsz_img_dir, img_filename)
        img_basename = osp.basename(img_filename).split('.')[0]
        
        img = sscv.imread(img_filepath)

        result = inference_segmentor(model, img_filepath)

        # read dummy psd file from directory 
        dummy_psd = PSDImage.open(dummy_psd_filepath)

        psd_layers = []

        bg_channels = [img[:,:,0], img[:,:,1], img[:,:,2]]
                
        bg_width, bg_height = img.shape[0], img.shape[1] 

        bg_layer = nested_layers.Image(name='Background',
                                    visible=True, opacity=255, group_id=0,
                                    blend_mode=enums.BlendMode.normal, 
                                    top=0, left=0, 
                                    bottom=bg_height, right=bg_width, channels=bg_channels,
                                    metadata=None, layer_color=0, color_mode=None)

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
    