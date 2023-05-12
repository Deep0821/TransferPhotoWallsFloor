import argparse
from mmseg.apis import inference_segmentor, init_segmentor
import os
import cv2
import numpy as np

cwd = os.path.dirname(os.path.abspath(__file__)) + '/'



def segmentation(img_path, options, out, look):
    """

        --img Image path
        --segm What parts to segment. Format --segm floor walls -> will segment walls and floor
        --out Output path

    """
    path = os.path.abspath(img_path)
    path_segment = cwd + "mmsegmentation/"
    config_file = path_segment + 'twins_svt-l_uperhead_8x2_512x512_160k_ade20k.py'
    checkpoint_file = path_segment + 'twins_svt-l_uperhead_8x2_512x512_160k_ade20k_20211130_141005-3e2cae61.pth'

    model = init_segmentor(config_file, checkpoint_file, device='cpu')
    img = cv2.imread(img_path)
    if look == "True":
        if img.shape[1] > 500:
            ratio = img.shape[0] / img.shape[1]
            cv2.imwrite(img_path.split(".")[0]+"_original.jpg", img)
            img = cv2.resize(img, (500, int(500 * ratio)))
            cv2.imwrite(img_path, img)
    else:
        if img.shape[1] > 2000:
            ratio = img.shape[0] / img.shape[1]
            cv2.imwrite(img_path.split(".")[0] + "_original.jpg", img)
            img = cv2.resize(img, (2000, int(2000 * ratio)))
            cv2.imwrite(img_path, img)

    result = inference_segmentor(model, path)[0]

    mask = np.full_like(result, False)
    for option in options:
        if option == "carpet":
            mask = mask | (result == 28) * 22
        elif option == "floor":
            mask = mask | (result == 3) * 22
        elif option == "walls":
            mask = mask | (result == 0) * 28

    #img = cv2.imread(path)
    #img = img * mask[..., np.newaxis]
    cv2.imwrite(out + "_mask.png", mask.astype(np.uint8))
    if look == "False":
        wall_mask = result == 0
        wall_mask = np.transpose([wall_mask, wall_mask, wall_mask], (1, 2, 0))
        only_wall = wall_mask * img
        wall_grayscale = cv2.cvtColor(only_wall, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(out + "_full_segmentation.png", result.astype(np.uint8))
        cv2.imwrite(out + "_wall_grayscale.png", wall_grayscale)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, required=True)
    parser.add_argument('--segm', choices=['carpet', 'floor', 'walls'], nargs="+", required=True)
    parser.add_argument('--look', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)

    args = parser.parse_args()

    img_path = args.img
    out = args.out
    options = args.segm
    look = args.look
    segmentation(img_path, options, out, look)
