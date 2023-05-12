import argparse
from mmseg.apis import inference_segmentor, init_segmentor
import os
import cv2
import numpy as np

cwd = os.path.dirname(os.path.abspath(__file__)) + '/'
def segmentation(img_path):
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
    result_original = inference_segmentor(model, path)[0]
    result_rescale = inference_segmentor(model, path, rescale=True)[0]
    print(result_rescala.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, required=True)


    args = parser.parse_args()

    img_path = args.img

    segmentation(img_path)
