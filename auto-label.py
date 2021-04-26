
import cv2
import matplotlib.pyplot as plt
import os.path as osp 
import sscv 

from glob import glob 
from tqdm import tqdm


print("Start Auto Labeling Procedure ....")

# Input directory information 
auto_label_dir = "/run/user/1000/gvfs/afp-volume:host=DiskStation.local,user=shm_nas,volume=Volume_2/06. Auto-Labeling/"

dataset_name = 'General Crack'
raw_img_dir_name = '/home/soojin/UOS-SSaS Dropbox/05. Data/06. Auto Labeling/General Crack/01. Raw Image'
rsz_img_dir_name = '02. Resized Image'
pre_label_dir_name = '03. Pre-labeled Psd Files'
corrected_label_dir_name = '04. Corrected Psd files' 
rsz_img_size = 2048

dataset_dir = osp.join(auto_label_dir, dataset_name)
raw_img_dir = osp.join(raw_img_dir_name)
rsz_img_dir = osp.join(dataset_dir, rsz_img_dir_name)
pre_label_dir = osp.join(dataset_dir, pre_label_dir_name)
corrected_label_dir = osp.join(dataset_dir, corrected_label_dir_name) 


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

    