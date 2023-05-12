import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps


path = "walls_mask_1616.jpg"
sat = cv2.imread("1616_sat.jpg")
cv2.imshow("", sat)
cv2.waitKey(0)

counts, bins, pathces = plt.hist(sat[sat != 0].flatten(), bins=60, density=True)
kde = sps.gaussian_kde(sat[sat != 0].flatten())
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


rt, thresh1 = cv2.threshold(sat, intervals[1][0], 255, cv2.THRESH_BINARY_INV)
rt, thresh2 = cv2.threshold(sat, intervals[1][1], 255, cv2.THRESH_BINARY_INV)
cv2.imshow("", thresh2 - thresh1)
cv2.waitKey(0)
cv2.imshow("", 255 - thresh2)
cv2.waitKey(0)
cv2.imshow("", thresh1)
cv2.waitKey(0)

