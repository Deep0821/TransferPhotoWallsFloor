import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps


def visualize(hsv_channel):
    counts, bins, pathces = plt.hist(hsv_channel[hsv_channel != 0].flatten(), bins=255, density=True)
    kde = sps.gaussian_kde(hsv_channel[hsv_channel != 0].flatten())
    x = np.linspace(0, 255, 301)
    mins = []
    last = kde.pdf(x[1]) - kde.pdf(x[0])
    for n in range(1, len(x) - 1):
        new_one = kde.pdf(x[n + 1]) - kde.pdf(x[n])
        if last * new_one < 0 and last < 0:
            mins.append(x[n])
        last = new_one
    print("mins", mins)
    intervals = [mins[k: k + 2] for k in range(len(mins) - 1)]
    print(intervals)
    plt.plot(x, kde.pdf(x), label='KDE')
    plt.show()


def shadows(walls_mask):
    gray = cv2.cvtColor(walls_mask, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(walls_mask, cv2.COLOR_BGR2HSV)
    mask = gray != 0
    only_hue = hsv[..., 0]
    only_sat = hsv[..., 1]
    only_value = hsv[..., 2]
    cv2.imwrite("value.jpg", only_value)
    cv2.imshow("", only_hue*mask)
    cv2.waitKey(0)
    cv2.imshow("", only_sat*mask)
    cv2.waitKey(0)
    cv2.imshow("", only_value*mask)
    cv2.waitKey(0)
    visualize(only_hue)
    visualize(only_sat)
    visualize(only_value)
    counts, bins, pathces = plt.hist(only_sat[only_sat != 0].flatten(), bins=10, density=True)
    kde = sps.gaussian_kde(only_sat[only_sat != 0].flatten())
    x = np.linspace(0, 255, 301)
    mins = []
    last = kde.pdf(x[1]) - kde.pdf(x[0])
    for n in range(1, len(x) - 1):
        new_one = kde.pdf(x[n + 1]) - kde.pdf(x[n])
        if last * new_one < 0 and last < 0:
            mins.append(x[n])
        last = new_one
    print("mins", mins)
    intervals = [mins[k: k + 2] for k in range(len(mins) - 1)]
    print(intervals)
    plt.plot(x, kde.pdf(x), label='KDE')
    plt.show()
    rt, thresh1 = cv2.threshold(only_sat, intervals[1][0], 255, cv2.THRESH_BINARY_INV)
    rt, thresh2 = cv2.threshold(only_sat, intervals[1][1], 255, cv2.THRESH_BINARY_INV)
    new = thresh2 - thresh1
    new = 255 - new

    new = new * mask
    new = new != 0
    new_img = walls_mask * new[..., np.newaxis]
    cv2.imwrite("shadow_remove_result_new.jpg", new_img.astype(np.uint8))
    cv2.imshow("", np.concatenate([walls_mask, new_img.astype(np.uint8)], axis=1))
    cv2.waitKey(0)

path = r"C:\Users\USER\Desktop\for_segmentation\home-design._mask.jpg"
walls_mask = cv2.imread(path)


shadows(walls_mask)