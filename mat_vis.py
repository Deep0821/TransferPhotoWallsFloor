import scipy.io as io
import numpy as np
from PIL import Image


def vis_mat(mat_img):
    mat = io.loadmat(mat_img)
    img_net = np.array([mat["layout"] for j in range(3)])
    img_net = np.transpose(img_net, (1, 2, 0))
    # img_net = (img_net - np.mean(img_net)) * 255 / (np.max(img_net) - np.min(img_net))
    img_net = img_net.astype(np.uint8)
    label_image = Image.fromarray(img_net)
    label_image.show()
    label_image.save("test_mat_image.jpg")

path = r"C:\Users\USER\Desktop\lsun_dataset_c\masks_mat\sun_agvbrhjzalwtdawo.mat"

vis_mat(path)