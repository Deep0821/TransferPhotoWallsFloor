import os
import shutil
path = r"C:\Users\USER\Desktop\lsun_dataset_c\surface_relabel"
save_path_imgs = r"C:\Users\USER\Desktop\lsun_dataset_c\images"
save_path_masks = r"D:\lsun_dataset_new\masks"
save_path_masks_mat = r"C:\Users\USER\Desktop\lsun_dataset_c\masks_mat"
for j in os.listdir(path):
    full_path = path + "\\" + j

    for k in os.listdir(full_path):
        if k[-3:] == "jpg":
            shutil.copy(full_path+"\\"+k, save_path_imgs)

        # elif k[-3:] == "png":
        #     shutil.copy(full_path+"\\"+k, save_path_masks)
        if k[-3:] == "mat":
            shutil.copy(full_path+"\\"+k, save_path_masks_mat)

