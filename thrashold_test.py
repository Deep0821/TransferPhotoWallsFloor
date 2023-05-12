import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps


def nothing(x):
    pass


def thrash(img):
    cv2.namedWindow('image')
    cv2.createTrackbar('tr1', 'image', 0, 255, nothing)

    # creating trackbars for Green color change
    cv2.createTrackbar('tr2', 'image', 0, 255, nothing)

    # creating trackbars for Blue color change

    img_new = img.copy()
    while (True):
        # show image
        cv2.imshow('image', np.concatenate([img, img_new], axis=1))

        # for button pressing and changing
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # get current positions of all Three trackbars
        tr1 = cv2.getTrackbarPos('tr1', 'image')
        tr2 = cv2.getTrackbarPos('tr2', 'image')


        # display color mixture
        rt, thresh1 = cv2.threshold(img, tr1, 255, cv2.THRESH_BINARY_INV)
        rt, thresh2 = cv2.threshold(img, tr2, 255, cv2.THRESH_BINARY_INV)
        img_new = (thresh2 - thresh1).astype(np.uint8)
    # close the window
    cv2.destroyAllWindows()

path = r"C:\Users\USER\Desktop\for_segmentation\home-design._mask.jpg"
walls_mask = cv2.imread(path)
hsv = cv2.cvtColor(walls_mask, cv2.COLOR_BGR2HSV)
walls_mask = hsv[..., 2]

thrash(walls_mask)