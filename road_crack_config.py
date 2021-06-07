configs = dict(
    auto_label_dir = "/home/sss/UOS-SSaS Dropbox/05. Data/06. Auto Labeling",
    dataset_name = 'Road Crack',
    rsz_img_size = 1024,

    model_config_file = '/home/sss/UOS-SSaS Dropbox/05. Data/03. Checkpoints/2020.09.02_deeplabv3plus_r101-d8_769x769_40k_concrete_damage_cs/deeplabv3plus_r101-d8_769x769_40k_concrete_damage_cs.py',
    model_checkpoint_file = '/home/sss/UOS-SSaS Dropbox/05. Data/03. Checkpoints/2020.09.02_deeplabv3plus_r101-d8_769x769_40k_concrete_damage_cs/iter_40000.pth',

    num_of_class = 1 , # not include background 
    erode_size = 2,

    palette = [
        [255, 0, 0], # crack 
    ], 

    classes = ['crack',], 
    )
    


