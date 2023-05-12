import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps


path = "one_wall_mask.jpg"
sat = cv2.imread("one_wall_mask.jpg")
cv2.imshow("", sat)
cv2.waitKey(0)
walls_mask = cv2.imread(path)

gray = cv2.cvtColor(walls_mask, cv2.COLOR_BGR2GRAY)
mask = gray != 0
hsv = cv2.cvtColor(walls_mask, cv2.COLOR_BGR2HSV)
cv2.imshow("", hsv[..., 0])
cv2.waitKey(0)
cv2.imshow("", hsv[..., 1])
cv2.waitKey(0)
cv2.imshow("", hsv[..., 2])
cv2.waitKey(0)
cv2.imwrite("1616_brightness.jpg", hsv[..., 2])
only_hue = hsv[..., 0]
only_sat = hsv[..., 1]
cv2.imwrite("sat_123.jpg", only_sat)
counts, bins, pathces = plt.hist(only_hue[only_hue != 0].flatten(), bins=40)
hue = bins[np.argmax(counts)]
print(hue)
print(np.mean(only_hue[only_hue != 0]))
mn_sat = np.mean(only_sat[only_sat != 0])
print(mn_sat)
print(np.std(only_sat[only_sat != 0]))
plt.show()
counts, bins, pathces = plt.hist(only_sat[only_sat != 0].flatten(), bins=60, density=True)
kde = sps.gaussian_kde(only_sat[only_sat != 0].flatten())
x = np.linspace(0, 255, 301)
mins = []
last = kde.pdf(x[1]) - kde.pdf(x[0])
for n in range(1, len(x) - 1):
    new_one = kde.pdf(x[n+1]) - kde.pdf(x[n])
    if last*new_one < 0 and last < 0:
        mins.append(x[n])
    last = new_one
print("mins", mins)
intervals = [mins[k: k+2] for k in range(len(mins)-1)]
print(intervals)
plt.plot(x, kde.pdf(x), label='KDE')
plt.show()
counts, bins, pathces = plt.hist(hsv[..., 2][hsv[..., 2] != 0].flatten(), bins=60, density=True)
plt.show()
most_comm = bins[np.argmax(counts)]
mn_hue = np.mean(only_hue[only_hue != 0])
rt, thresh1 = cv2.threshold(only_sat, intervals[1][0], 255, cv2.THRESH_BINARY_INV)
rt, thresh2 = cv2.threshold(only_sat, intervals[1][1], 255, cv2.THRESH_BINARY_INV)
new = thresh2 - thresh1
new = 255-new

cv2.imshow("", new)
cv2.waitKey(0)
new = new*mask

cv2.imshow("", new*mask)
cv2.waitKey(0)
new = new != 0
new_img = walls_mask * new[..., np.newaxis]
print(new_img.shape)
cv2.imwrite("shadow_remove_res.jpg", new_img.astype(np.uint8))
cv2.imshow("", new_img.astype(np.uint8))
cv2.waitKey(0)
cv2.imshow("", thresh2 - thresh1)
cv2.waitKey(0)
cv2.imshow("", 255 - thresh2)
cv2.waitKey(0)
cv2.imshow("", thresh1)
cv2.waitKey(0)
cv2.imshow("", only_hue)
cv2.waitKey(0)
cv2.imshow("", hsv[..., 1])
cv2.waitKey(0)
cv2.imwrite("sat.jpg", hsv[..., 1])


