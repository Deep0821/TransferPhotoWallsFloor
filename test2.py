import cv2
import numpy as np

mask = np.zeros((1000, 1000, 3))
points = np.array([[100, 100], [100, 200], [200, 100], [200, 200]])
mask = cv2.polylines(mask, [points.reshape((-1, 1, 2))], True, (255, 25, 100), 3)
cv2.imshow("", mask)
cv2.waitKey(0)
# cv2.imwrite("new_result.jpg", full_img)
#     cv2.imshow("", full_img)
#     cv2.waitKey(0)
#     left_wall = hsv[:, :480]
#     right_wall = hsv[:, 480:920]
#     l = left_wall != 0
#     right_mean_v = np.mean(right_wall[right_wall != 0])
#     left_mean_v = np.mean(left_wall[l])
#
#     left_wall = left_wall.astype(np.float64)
#     left_wall[l] = ((left_wall[l] - left_mean_v + right_mean_v))
#     left_wall = ((left_wall * 255 / np.max(left_wall)).astype(np.uint8))
#     new_image = np.concatenate((left_wall, right_wall), axis=1)
#     cv2.imshow("", new_image)
#     cv2.waitKey(0)

#
# img = img.astype(np.float64)
#     l = img != 0
#     orig_mean = np.mean(img[l])
#     img[l] = img[l] - orig_mean + mean
#     img[l] = (img[l]/std)*np.std(img[l])
#     # img[l] = img[l] * (std / np.std(img[l]))
#     img = np.clip(img, 0, 255)
#
#     img = img.astype(np.uint8)

# def change_color_mean(img, mean, std):
#     img = img.astype(np.float64)
#     l = img != 0
#     orig_mean = np.mean(img[l])
#     img[l] = img[l] - orig_mean
#     img[l] = (img[l]*std)/np.std(img[l])
#     img[l] = img[l] + mean
#     # img[l] = img[l] * (std / np.std(img[l]))
#     img = np.clip(img, 0, 255)
#
#     img = img.astype(np.uint8)
#     return img

def wall_same_light(img, corner_points):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    step = len(corner_points)//2
    mask = np.zeros(hsv.shape, dtype=np.uint8)
    change_points = corner_points[[0, 1, step+1, step]]
    if change_points[0, 0] > 0:
        change_points = np.insert(change_points, 0, [0, 0], axis=0)
    if change_points[-1, 0] > 0:
        change_points = np.insert(change_points, len(change_points) - 1, [0, hsv.shape[0]], axis=0)
    change_points = change_points.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [change_points], 1)
    cv2.imwrite("msk.jpg", mask)

    img_new = mask*hsv
    change_mean = np.mean(img_new[img_new != 0])
    change_std = np.std(img_new[img_new != 0])
    full_img = img_new.copy()
    for num in range(1, step-1):
        mask = np.zeros(hsv.shape)
        four_points = corner_points[[num, num+1, num+step+1, num+step]]
        if num == step-2:
            if four_points[1, 0] < hsv.shape[1]:
                four_points = np.insert(four_points, 2, [hsv.shape[1], 0], axis=0)
            if four_points[-2, 0] < hsv.shape[1]:
                four_points = np.insert(four_points, len(four_points)-2, [hsv.shape[1], hsv.shape[0]], axis=0)
        four_points = four_points.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [four_points], 1)
        img_new = mask*hsv
        img_new = change_color_mean(img_new, change_mean, change_std)
        full_img += img_new

    new_img = (img != 0)*1
    new_img_1 = new_img*np.full(img.shape, [230, 196, 78])
    print("color", new_img_1)
    new_img_1 = cv2.cvtColor(new_img_1.astype(np.uint8), cv2.COLOR_RGB2HSV)
    new_img_1 = new_img_1.astype(np.float64)
    full_img = full_img[..., np.newaxis]
    new_full = np.concatenate([full_img, full_img, full_img], axis=2)
    print(new_full)

    cv2.imshow("", new_img_1.astype(np.uint8))
    cv2.waitKey(0)
    cv2.imshow("", cv2.cvtColor(new_img_1.astype(np.uint8), cv2.COLOR_HSV2RGB))
    cv2.waitKey(0)
    # change.level(black=-0.5, white=1, channel='all_channels')
    new_img_1[..., 2] = (full_img[..., 0] * new_img_1[..., 2] / 255)
    new_img_1[..., 2] *= 1.9
    new_result_img = np.clip(new_img_1, 0, 255)
    print(new_result_img)
    cv2.imshow("", new_result_img.astype(np.uint8))
    cv2.waitKey(0)

    new_result_img = cv2.cvtColor(new_result_img.astype(np.uint8), cv2.COLOR_HSV2RGB)
    print(new_result_img)
    cv2.imshow("", new_result_img.astype(np.uint8))
    cv2.waitKey(0)

    hsv = hsv[..., np.newaxis]
    cv2.imwrite("final_result_4.jpg", new_result_img)
    cv2.imwrite("without_algo.jpg", (hsv*new_img_1/255).astype(np.uint8))
    assert 0
    return new_img
