import cv2
from wand.image import Image
import os
from mmseg.apis import inference_segmentor, init_segmentor
import numpy as np

colors = [[235,233,245],
[219,236,248],
[240,238,226],
[218,228,224],
[197,205,235],
[177,224,241],
[197,223,212],
[231,222,190],
[155,133,183],
[124,175,219],
[123,173,166],
[159,168,124],
[172,148,117],
[90,57,133],
[57,92,181],
[72,156,214],
[58,121,110],
[70,102,45],
[111,75,33],
[53,43,100],
[48,104,143],
[39,81,73],
[64,70,31],
[74,51,22],
[21,26,57],
[22,35,45],
[44,31,13],
[201,201,198],
[170,209,224],
[160,138,157],
[138,136,133],
[122,151,173],
[93,58,86],
[120,108,93],
[54,138,153],
[228,221,217],
[66,78,170],
[54,138,153],
[86,71,32],
[155,133,183]]


cwd = os.path.dirname(os.path.abspath(__file__)) + '/'

path_segment = cwd + "mmsegmentation/"
config_file = path_segment + 'twins_svt-l_uperhead_8x2_512x512_160k_ade20k.py'
checkpoint_file = path_segment + 'twins_svt-l_uperhead_8x2_512x512_160k_ade20k_20211130_141005-3e2cae61.pth'

model = init_segmentor(config_file, checkpoint_file, device='cpu')

def transport_shadows(img1, img2):
    img1 = Image.from_array(img1)
    img2 = Image.from_array(img2)
    img1.transform_colorspace('gray')

    img1.level(black=-0.5, white=1, channel='all_channels')

    img2.composite_channel('all_channels', img1, 'hard_light', 0, 0)
    img2 = np.array(img2)
    return img2

def mask2image(mask, image_orig, image_transform):
    image_orig[mask] = [0, 0, 0]
    image_orig += image_transform
    image_orig.astype(np.uint8)
    return image_orig

for path in os.listdir("img_path"):
    full_path = "img_path" + "\\" +path
    img = cv2.imread(full_path)
    result = inference_segmentor(model, full_path)[0]
    result_mask_wall = result == 0
    result_mask_3d = np.array([result_mask_wall for _ in range(3)])
    result_mask_3d = np.transpose(result_mask_3d, (1, 2, 0))

    only_wall_orig = img * result_mask_3d
    # cv2.imshow("", only_wall_orig.astype(np.uint8))
    # cv2.waitKey(0
    for rgb in colors:
        rgb_image = np.full_like(img, rgb[::-1])
        new_wall = transport_shadows(only_wall_orig, rgb_image)
        new_wall = new_wall * result_mask_3d

        img = mask2image(result_mask_wall, img, new_wall)
        cv2.imwrite("save_path_new1/"+path[:-4] + "_" +str(rgb) + ".jpg", img)

