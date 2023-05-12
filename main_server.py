import cv2
import numpy as np
import torch
from datasets import sequence
from trainer import core
from mmseg.apis import inference_segmentor, init_segmentor
from wand.image import Image as Image1
import argparse
from PIL import Image
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

from wand.display import display
import os
cwd = os.path.dirname(os.path.abspath(__file__)) + '/'
tiles_folder = "/root/floor_swap/floor_module/floor_material/"
# torch.backends.cudnn.benchmark = True
# torch.cuda.empty_cache()


class Predictor:
    def __init__(self, weight_path):
        self.model = core.LayoutSeg.load_from_checkpoint(weight_path, backbone='resnet101')

        self.model.freeze()
        self.model.eval()

    @torch.no_grad()
    def feed(self, image: torch.Tensor) -> np.ndarray:

        self.model.to(torch.device("cpu"))
        _, outputs = self.model(image[None, :, :, :].cpu())

        label = core.label_as_rgb_visual(outputs.cpu()).squeeze(0)
        label = label.cpu()
        return label.permute(1, 2, 0).numpy()


def isnt_duplicate(check_point, anchor_points, criterion=20):
    check_point = np.array(check_point)
    for anchor_point in anchor_points:
        anchor_point = np.array(anchor_point)
        if np.all((check_point < anchor_point+criterion) & (check_point > anchor_point-criterion)):
            return False
            break
    else:
        return True


def main1(path, weight, image_size):  # Predict
    predictor = Predictor(weight_path=weight)
    image = Image.open(path).convert('RGB')
    shape = image.size

    image = F.resize(image, (image_size, image_size), interpolation=Image.BILINEAR)
    image = F.to_tensor(image)
    image = F.normalize(image, mean=0.5, std=0.5)
    # image = image.test()
    return predictor.feed(image)


def find_k_and_b(first_point, second_point):
    k = (first_point[1] - second_point[1]) / (first_point[0] - second_point[0])
    b = -second_point[0] * (first_point[1] - second_point[1]) / (first_point[0] - second_point[0]) + second_point[1]
    return k, b


def two_line_intersection(line1_params, line2_params):
    (k1, b1), (k2, b2) = line1_params, line2_params
    x = (b2 - b1) / (k1 - k2)
    y = k2 * x + b2
    return x, y


def corners_with_color(prediction, img_size):
    corner_points = []
    matrix = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]])

    # cv2.filter2d()
    for row in range(1, img_size - 1):
        for column in range(1, img_size - 1):
            if len(np.unique(prediction[row - 1:row + 2, column - 1:column + 2] * matrix)) == 4:
                if isnt_duplicate([row, column], corner_points):
                    corner_points.append([row, column])
    return np.array(corner_points)


def transport_shadows(img1, img2):

    img1 = Image1.from_array(img1)
    img2 = Image1.from_array(img2)
    img1.transform_colorspace('gray')

    img1.level(black=-0.5, white=1, channel='all_channels')

    img2.composite_channel('all_channels', img1, 'hard_light', 0, 0)
    img2 = np.array(img2)

    return img2


def extract_new_points(floor_segment):
    vertical_point_count = np.array([np.sum(floor_segment[row]) for row in range(floor_segment.shape[0])])
    horizontal_point_count = np.array([np.sum(floor_segment[:, column]) for column in range(floor_segment.shape[1])])
    min_height = np.argwhere(vertical_point_count != 0)[0]


    min_width, max_width = np.argwhere(horizontal_point_count != 0)[[0, -1]]
    min_width = min_width - 1 if min_width > 0 else min_width
    max_width = max_width + 1 if max_width < floor_segment.shape[1] else max_width
    min_x = np.argwhere(floor_segment[min_height] != 0)[0, 1]
    min_height[0] = min_height[0] - 2
    left_point = [min_width, min_height]
    right_point = [max_width, min_height]
    max_point = [min_x, min_height]
    return left_point, right_point, max_point

def one_vanishing_point_case(vanishing_point_1, floor_mask):
    left_point, right_point, max_point = extract_new_points(floor_mask)




def black_cases_2_corner(vanishing_point_1, vanishing_point_2, floor_mask):

    left_point, right_point, max_point = extract_new_points(floor_mask)
    if vanishing_point_1[0] > vanishing_point_2[0]:

        bottom_point = [0, floor_mask.shape[0]]


        vanishing_1_left = find_k_and_b(left_point, vanishing_point_1)
        vanishing_1_right = find_k_and_b(right_point, vanishing_point_1)

        vanishing_2_bottom = find_k_and_b(vanishing_point_2, bottom_point)
        vanishing_2_right = find_k_and_b(vanishing_point_2, right_point)
        point_1 = two_line_intersection(vanishing_1_left, vanishing_2_bottom)
        point_2 = two_line_intersection(vanishing_2_right, vanishing_1_left)
        point_3 = two_line_intersection(vanishing_1_right, vanishing_2_bottom)
        point_4 = right_point

        perspective_points = []
        perspective_points.append(list(point_1))
        perspective_points.append(list(point_2))
        perspective_points.append(list(point_3))
        perspective_points.append(list(point_4))
    else:

        bottom_point = list(floor_mask[::-1].shape)

        vanishing_1_left = find_k_and_b(left_point, vanishing_point_1)
        vanishing_1_right = find_k_and_b(right_point, vanishing_point_1)

        vanishing_2_bottom = find_k_and_b(vanishing_point_2, bottom_point)
        vanishing_2_left = find_k_and_b(vanishing_point_2, left_point)
        point_1 = two_line_intersection(vanishing_1_right, vanishing_2_left)
        point_2 = two_line_intersection(vanishing_1_right, vanishing_2_bottom)
        point_4 = two_line_intersection(vanishing_1_left, vanishing_2_bottom)
        point_3 = left_point

        perspective_points = []
        perspective_points.append(list(point_1))
        perspective_points.append(list(point_2))
        perspective_points.append(list(point_3))
        perspective_points.append(list(point_4))

    return perspective_points

def black_cases_1_corner(vanishing_point_1, vanishing_point_2, floor_mask):
    v_points = np.array([vanishing_point_1, vanishing_point_2])
    sort_args = np.argsort(v_points[:, 0])


    vanishing_point_1, vanishing_point_2 = v_points[sort_args]

    left_point, right_point, max_point = extract_new_points(floor_mask)
    if vanishing_point_1[0] > 0:
        bottom_point_left = [0, left_point[1]]
        vanishing_1_left = find_k_and_b(bottom_point_left, vanishing_point_1)
        vanishing_1_right = find_k_and_b(right_point, vanishing_point_1)
    else:
        bottom_point_left = [0, floor_mask.shape[0]]
        vanishing_1_right = find_k_and_b(right_point, vanishing_point_1)
        vanishing_1_left = find_k_and_b(bottom_point_left, vanishing_point_1)

    if vanishing_point_2[0] <= floor_mask.shape[1]:
        print("is")
        bottom_point_right = [floor_mask.shape[1], right_point[1]]
        vanishing_2_right = find_k_and_b(bottom_point_right, vanishing_point_2)
        vanishing_2_left = find_k_and_b(left_point, vanishing_point_2)
    else:
        bottom_point_right = list(floor_mask.shape[::-1])
        vanishing_2_left = find_k_and_b(vanishing_point_2, left_point)
        vanishing_2_right = find_k_and_b(vanishing_point_2, bottom_point_right)


    point_1 = two_line_intersection(vanishing_1_right, vanishing_2_left)
    point_2 = two_line_intersection(vanishing_1_right, vanishing_2_right)
    point_3 = two_line_intersection(vanishing_1_left, vanishing_2_left)
    point_4 = two_line_intersection(vanishing_1_left, vanishing_2_right)

    perspective_points = []
    perspective_points.append(list(point_1))
    perspective_points.append(list(point_2))
    perspective_points.append(list(point_3))
    perspective_points.append(list(point_4))

    return perspective_points

def mask2image(mask, image_orig, image_transform):
    image_orig[mask] = [0, 0, 0]
    image_orig += image_transform
    image_orig.astype(np.uint8)
    return image_orig


def default_perspective(floor_segment):
    left_point, right_point, max_point = extract_new_points(floor_segment)
    left_point[0] = list(left_point[0])[0]
    left_point[1] = list(left_point[1])[0]
    right_point[0] = list(right_point[0])[0]
    right_point[1] = list(right_point[1])[0]

    vanishing_point = [floor_segment.shape[1]//2, left_point[1]-200]
    left_line = find_k_and_b(vanishing_point, left_point)
    right_line = find_k_and_b(vanishing_point, right_point)
    right_line_inter = [(floor_segment.shape[0] - right_line[1])/right_line[0], floor_segment.shape[0]]
    left_line_inter = [(floor_segment.shape[0] - left_line[1])/left_line[0], floor_segment.shape[0]]
    pts = [left_point, right_point, left_line_inter, right_line_inter]
    return pts


def no_corner_vertical(floor_segment):
    print("vertical")
    left_point, right_point, max_point = extract_new_points(floor_segment)
    vanishing_point = [floor_segment.shape[1]//2, left_point[1]-200]
    print(vanishing_point)
    left_line = find_k_and_b(left_point, vanishing_point)
    right_line = find_k_and_b(right_point, vanishing_point)

    point_1 = left_point
    point_2 = right_point
    point_3 = [(floor_segment.shape[0] - left_line[1])/left_line[0], floor_segment.shape[0]]
    point_4 = [(floor_segment.shape[0] - right_line[1]) / right_line[0], floor_segment.shape[0]]

    corner_points = []
    corner_points.append(point_1)
    corner_points.append(point_2)
    corner_points.append(point_3)
    corner_points.append(point_4)

    return corner_points


def one_corner_case(prediction, corner_points, height, width, img_size):
    print("one corner")


    edges = cv2.Canny(prediction, 20, 200)

    edges_upper = edges.copy()
    edges_bottom = edges.copy()
    edges_upper[height//2:] = 0
    edges_bottom[:height//2] = 0

    corner_points = corner_points[:, ::-1]

    corner_points_old = []
    have_upper = np.array([np.sum(edges_upper[:, column]) for column in range(width)])
    have_bottom = np.array([np.sum(edges_bottom[:, column]) for column in range(width)])
    left_upper_line_x, right_upper_line_x = np.argwhere(have_upper != 0)[[0, -1]]
    left_line_x, right_line_x = np.argwhere(have_bottom != 0)[[0, -1]]

    left_max = [left_upper_line_x[0], np.argwhere(edges[:, left_upper_line_x] != 0)[0, 0]]
    right_max = [right_upper_line_x[0], np.argwhere(edges[:, right_upper_line_x] != 0)[0, 0]]
    corner_points_old.append(left_max)


    corner_points_old.append(corner_points[0])
    corner_points_old.append(right_max)
    right_upper_line_k, right_upper_line_b = find_k_and_b(corner_points[0], right_max)

    left_min = [left_line_x[0], np.argwhere(edges_bottom[:, left_line_x] != 0)[-1, 0]]
    right_min = [right_line_x[0], np.argwhere(edges_bottom[:, right_line_x] != 0)[-1, 0]]

    upper_line_k, upper_line_b = find_k_and_b(left_max, corner_points[0])
    corner_points = corner_points[-1]
    corner_points_old.append(left_min)
    corner_points_old.append(corner_points)

    corner_points_old.append(right_min)
    left_line_k, left_line_b = find_k_and_b(corner_points, left_min)
    right_line_k, right_line_b = find_k_and_b(corner_points, right_min)

    right_line_inter_x, right_line_inter_y = two_line_intersection([right_line_k, right_line_b], [right_upper_line_k, right_upper_line_b])
    vanishing_points = []
    vanishing_points.append([right_line_inter_x, right_line_inter_y])

    # left_line_inter_bottom_x = (img_size - left_line_b)/left_line_k
    # choose_point_left = [left_line_inter_bottom_x, img_size] if right_line_inter_x < 0 else left_min

    choose_point_left = [0, img_size] if left_min[0] < 2 else left_min

    new_line_k, new_line_b = find_k_and_b([right_line_inter_x, right_line_inter_y], choose_point_left)

    first_point = two_line_intersection([new_line_k, new_line_b], [left_line_k, left_line_b])

    point_2 = left_min
    point_3 = [(img_size-new_line_b)/new_line_k, img_size]
    point_4 = [img_size, img_size]
    vanishing_and_right_x, vanishing_and_right_y = (right_line_inter_y-left_line_b)/left_line_k, right_line_inter_y
    vanishing_points.append([vanishing_and_right_x, vanishing_and_right_y])

    point_4 = right_min if right_min[0] < img_size-1 else point_4

    right_new_line_k, right_new_line_b = find_k_and_b([vanishing_and_right_x, vanishing_and_right_y], point_4)
    point_4 = two_line_intersection([right_new_line_k, right_new_line_b], [right_line_k, right_line_b])
    need_point = two_line_intersection([right_new_line_k, right_new_line_b], [new_line_k, new_line_b])

    corner_new_points = [first_point]
    corner_new_points.append(corner_points)
    corner_new_points.append(need_point)
    corner_new_points.append(point_4)
    corner_new_points = np.array(corner_new_points)
    return corner_new_points, np.array(corner_points_old), vanishing_points, "one_corner"


def one_corner_without_ceiling(prediction, corner_points, height, width, img_size):

    edges = cv2.Canny(prediction, 20, 200)
    edges_bottom = edges.copy()
    edges_bottom[:height//2] = 0

    corner_points = corner_points[:, ::-1]
    corner_points = corner_points[0]

    have_bottom = np.array([np.sum(edges_bottom[:, column]) for column in range(width)])
    left_line_x, right_line_x = np.argwhere(have_bottom != 0)[[0, -1]]

    left_min = [left_line_x[0], np.argwhere(edges_bottom[:, left_line_x] != 0)[-1, 0]]
    right_min = [right_line_x[0], np.argwhere(edges_bottom[:, right_line_x] != 0)[-1, 0]]


    left_bottom_line = find_k_and_b(left_min, corner_points)
    right_bottom_line = find_k_and_b(right_min, corner_points)

    vanishing_point_right = (height//2 - left_bottom_line[1])/left_bottom_line[0]
    vanishing_point_left = (height//2 - right_bottom_line[1])/left_bottom_line[0]

    vanishing_points = np.array([[vanishing_point_left, height//2], [vanishing_point_right, height//2]])
    corner_old = 0
    return np.array([corner_points]).astype(np.float32), np.array(corner_old), vanishing_points, "one_corner_case_without_ceiling"

def two_corner_case(prediction, corner_points, height, width):
    # Find image-border points
    print("two_corner")

    edges = cv2.Canny(prediction, 20, 200)

    edges_upper = edges.copy()
    edges_bottom = edges.copy()
    edges_upper[height // 2:] = 0

    edges_bottom[:height // 2] = 0
    edges[:height//2] = 0
    corner_points_old = []
    have_upper = np.array([np.sum(edges_upper[:, column]) for column in range(width)])
    left_upper_line_x, right_upper_line_x = np.argwhere(have_upper != 0)[[0, -1]]

    left_max = [left_upper_line_x[0], np.argwhere(edges_upper[:, left_upper_line_x] != 0)[0, 0]]
    right_max = [right_upper_line_x[0], np.argwhere(edges_upper[:, right_upper_line_x] != 0)[0, 0]]
    if corner_points[0, 1] < corner_points[1, 1]:
        corner_points_old.append(left_max)
        corner_points_old.append(list(corner_points[0, ::-1]))
        corner_points_old.append(list(corner_points[1, ::-1]))
        corner_points_old.append(right_max)
    else:
        corner_points_old.append(left_max)
        corner_points_old.append(list(corner_points[1, ::-1]))
        corner_points_old.append(list(corner_points[0, ::-1]))
        corner_points_old.append(right_max)

    have = np.array([np.sum(edges[:, column]) for column in range(width)])  # Write with simple loop

    left_line_x, right_line_x = np.argwhere(have != 0)[[0, -1]]

    # Reverse corner points
    corner_points = corner_points[-2:]

    widths = corner_points[:, 1]
    indexes = np.argsort(widths)
    corner_points = corner_points[indexes]

    left_min = [left_line_x[0], np.argwhere(edges[:, left_line_x] != 0)[-1, 0]]
    right_min = [right_line_x[0], np.argwhere(edges[:, right_line_x] != 0)[-1, 0]]
    corner_points = corner_points[:, ::-1]

    if corner_points[0, 0] < corner_points[1, 0]:
        corner_points_old.append(left_min)
        corner_points_old.append(list(corner_points[0]))
        corner_points_old.append(list(corner_points[1]))
        corner_points_old.append(right_min)
    else:
        corner_points_old.append(left_min)
        corner_points_old.append(list(corner_points[1]))
        corner_points_old.append(list(corner_points[0]))
        corner_points_old.append(right_min)

    # Find slope and bias for 3 lines

    left_line_k, left_line_b = find_k_and_b(corner_points[0], left_min)
    right_line_k, right_line_b = find_k_and_b(corner_points[1], right_min)
    upper_line_k, upper_line_b = find_k_and_b(corner_points[0], corner_points[1])

    # Find outer scope points
    x1 = (height-left_line_b) / left_line_k
    x2 = (height-right_line_b) / right_line_k


    # Determine room case
    x_choose = x2 if corner_points[0][1] > corner_points[1][1] else x1
    lines_choose_k, lines_choose_b = (left_line_k, left_line_b)

    if corner_points[0][1] < corner_points[1][1]:
        print("right")
        lines_choose_k, lines_choose_b = (right_line_k, right_line_b)

    # Find vanishing point
    vanishing_point_x, vanishing_point_y = two_line_intersection([left_line_k, left_line_b], [right_line_k, right_line_b])
    vanishing_points = [[vanishing_point_x, vanishing_point_y]]

    upper_line_vanishing_point = (vanishing_point_y - upper_line_b) / upper_line_k
    vanishing_points.append([upper_line_vanishing_point, vanishing_point_y])

    bottom_line_k, bottom_line_b = find_k_and_b([upper_line_vanishing_point, vanishing_point_y], [x_choose, height])
    # Use two_line_intersection function
    x_intersection = (bottom_line_b - lines_choose_b) / (lines_choose_k - bottom_line_k)
    y_intersection = bottom_line_k * x_intersection + bottom_line_b

    if x_choose == x1:
        corner_points = np.append(corner_points, [[x_choose, height]], axis=0)
        corner_points = np.append(corner_points, [[x_intersection, y_intersection]], axis=0)
    else:

        corner_points = np.append(corner_points, [[x_intersection, y_intersection]], axis=0)
        corner_points = np.append(corner_points, [[x_choose, height]], axis=0)


    return corner_points, np.array(corner_points_old, dtype=np.float32), vanishing_points, "two_corner"


def no_corner_case(prediction, height, width, img_size):
    edges = cv2.Canny(prediction, 20, 200)
    edges_upper = edges.copy()
    edges_bottom = edges.copy()
    edges_bottom[:height // 2] = 0
    edges_upper[height // 2:] = 0

    have_upper = np.array([np.sum(edges_upper[:, column]) for column in range(width)])
    have_bottom = np.array([np.sum(edges_bottom[:, column]) for column in range(width)])
    if len(np.argwhere(have_upper != 0)) != 0 and len(np.argwhere(have_bottom != 0)):
        print("no_corner_with_ceiling")
        have_bottom = np.array([np.sum(edges_bottom[:, column]) for column in range(width)])
        print(have_bottom)
        left_upper_line_x, right_upper_line_x = np.argwhere(have_upper != 0)[[0, -1]]
        left_line_x, right_line_x = np.argwhere(have_bottom != 0)[[0, -1]]

        left_max = [left_upper_line_x[0], np.argwhere(edges[:, left_upper_line_x] != 0)[0, 0]]
        right_max = [right_upper_line_x[0], np.argwhere(edges[:, right_upper_line_x] != 0)[0, 0]]

        left_min = [left_line_x[0], np.argwhere(edges_bottom[:, left_line_x] != 0)[-1, 0]]
        right_min = [right_line_x[0], np.argwhere(edges_bottom[:, right_line_x] != 0)[-1, 0]]

        line_k, line_b = find_k_and_b(left_min, right_min)
        upper_line_k, upper_line_b = find_k_and_b(left_max, right_max)
        vanishing_point_1 = two_line_intersection((line_k, line_b), (upper_line_k, upper_line_b))
        vanishing_point_2 = [-(width - vanishing_point_1[0]), vanishing_point_1[1]] if vanishing_point_1[0] < width else [
            -vanishing_point_1[0], vanishing_point_1[1]]
        vanishing_points = np.array([vanishing_point_1, vanishing_point_2])

    else:
        print("no_corner_no_ceiling")
        have_bottom = np.array([np.sum(edges_bottom[:, column]) for column in range(width)])
        left_line_x, right_line_x = np.argwhere(have_upper != 0)[[0, -1]] if len(np.argwhere(have_bottom != 0)) == 0 else np.argwhere(have_bottom != 0)[[0, -1]]
        edges_bottom = edges_upper if len(np.argwhere(have_bottom != 0)) == 0 else edges_bottom
        print(have_bottom)
        left_min = [left_line_x[0], np.argwhere(edges_bottom[:, left_line_x] != 0)[-1, 0]]
        right_min = [right_line_x[0], np.argwhere(edges_bottom[:, right_line_x] != 0)[-1, 0]]
        line_k, line_b = find_k_and_b(left_min, right_min)
        vanishing_point_1 = [(height//2-line_b)/line_k, height//2]
        vanishing_point_2 = [-(width - vanishing_point_1[0]), vanishing_point_1[1]] if vanishing_point_1[0] < width else [
            -vanishing_point_1[0], vanishing_point_1[1]]
        vanishing_points = np.array([vanishing_point_1, vanishing_point_2]).astype(np.float64)

    if np.arctan(line_k) < 4 * np.pi/180 and np.arctan(line_k) > -4 * np.pi/180:
        print("is vertical")
        central_point = [round(width//2), round(height//2)]
        new_right_k, new_right_b = find_k_and_b(central_point, right_min)
        new_left_k, new_left_b = find_k_and_b(central_point, left_min)
        x_left = (height - new_left_b)/new_left_k
        x_right = (height - new_right_b) / new_right_k
        corner_points = [left_min, right_min, [x_left, height], [x_right, height]]
        return np.array(corner_points), np.array([0]), np.array([[1, 2], [3, 4]]).astype(np.float64), "no_corner_vertical"

    # upper_line_k, upper_line_b = find_k_and_b(left_max, right_max)
    # vanishing_point_1 = two_line_intersection((line_k, line_b), (upper_line_k, upper_line_b))
    # vanishing_point_2 = [-(width-vanishing_point_1[0]), vanishing_point_1[1]] if vanishing_point_1 < width else [-vanishing_point_1[0], vanishing_point_1[1]]
    # vanishing_points = np.array([vanishing_point_1, vanishing_point_2])

    # if left_min[1] < right_min[1]:
    #
    #     bottom_line_k, bottom_line_b = find_k_and_b([0, height], [inter_x, inter_y])
    #     vanishing_point = [width - round(width / 4), inter_y]
    #     left_new_line_k, left_new_line_b = find_k_and_b(vanishing_point, left_min)
    #     right_new_line_k, right_new_line_b = find_k_and_b(vanishing_point, right_min)
    #     bottom_left_inter_x, bottom_left_inter_y = two_line_intersection((left_new_line_k, left_new_line_b), (bottom_line_k, bottom_line_b))
    #     bottom_right_inter_x, bottom_right_inter_y = two_line_intersection((right_new_line_k, right_new_line_b),
    #                                                                       (bottom_line_k, bottom_line_b))
    #
    #     corner_points = [left_min, (bottom_left_inter_x, bottom_left_inter_y), right_min, (bottom_right_inter_x, bottom_right_inter_y)]
    #
    # elif left_min[1] > right_min[1]:
    #
    #     bottom_line_k, bottom_line_b = find_k_and_b([width, height], [inter_x, inter_y])
    #     vanishing_point = [round(width/4), inter_y]
    #     right_new_line_k, right_new_line_b = find_k_and_b(vanishing_point, right_min)
    #     left_new_line_k, left_new_line_b = find_k_and_b(vanishing_point, left_min)
    #     bottom_right_inter_x, bottom_right_inter_y = two_line_intersection((right_new_line_k, right_new_line_b), (bottom_line_k, bottom_line_b))
    #     bottom_left_inter_x, bottom_left_inter_y = two_line_intersection((left_new_line_k, left_new_line_b),
    #                                                                      (bottom_line_k, bottom_line_b))
    #     corner_points = [(bottom_right_inter_x, bottom_right_inter_y), right_min, (bottom_left_inter_x, bottom_left_inter_y), left_min]
    corner_points = [[1, 2], [3, 4]]
    return np.array(corner_points).astype(np.float64), np.array([left_min, right_min, left_min, right_min]), vanishing_points, "no_corner"


def lightest_pixel(distance, img_shape):
    distance_height, distance_width = distance.shape
    distance = np.array(distance)

    brightness = [0]*img_shape[1]
    rows = [0]*img_shape[1]
    for column in range(img_shape[1], distance_width-img_shape[1]):
        z = distance[:, column]
        z[:img_shape[0]] = 0
        z[distance_height - img_shape[0]:] = 0

        rows.append(np.argmax(z))
        brightness.append(np.max(z))

    brightness = np.array(brightness)
    column = np.argsort(brightness)[-1]

    row = rows[column]

    return row, column


def add_logo(image, floor_mask, wall_mask, floor_image_rgba, wall_image_rgba):
    floor_image = floor_image_rgba[..., :3]
    wall_image = wall_image_rgba[..., :3]
    floor_rgba_mask = floor_image_rgba[..., -1]
    wall_rgba_mask = wall_image_rgba[..., -1]

    for o in range(3):
        floor_image[..., o] *= (floor_rgba_mask == 0) | (floor_rgba_mask == 255)
        wall_image[..., o] *= (wall_rgba_mask == 0) | (wall_rgba_mask == 255)

    floor_mask = floor_mask*255
    wall_mask = wall_mask*255

    image_size_floor = 100
    image_size_wall = 70
    floor_mask = floor_mask.astype(np.uint8)
    wall_mask = wall_mask.astype(np.uint8)
    floor_distance = cv2.distanceTransform(floor_mask, cv2.DIST_L2, 3)
    wall_distance = cv2.distanceTransform(wall_mask, cv2.DIST_L2, 3)
    image_ratio_floor = floor_image.shape[0]/floor_image.shape[1]
    image_ratio_wall = wall_image.shape[0] / wall_image.shape[1]
    image_size_floor = (round(image_size_floor * (1 / image_ratio_floor)), image_size_floor)
    image_size_wall = (round(image_size_wall * (1 / image_ratio_wall)), image_size_wall)

    floor_image = cv2.resize(floor_image, image_size_floor, 0)
    wall_image = cv2.resize(wall_image, image_size_wall, 0)


    row_wall, column_wall = lightest_pixel(wall_distance, wall_image.shape)
    row_floor, column_floor = lightest_pixel(floor_distance, floor_image.shape)

    image_copy = image.copy()
    image[row_floor:row_floor + image_size_floor[1], column_floor:column_floor + image_size_floor[0]] = floor_image
    image[row_wall:row_wall + image_size_wall[1], column_wall:column_wall + image_size_wall[0]] = wall_image
    mask = image == [0, 0, 0]

    new_image = mask * image_copy

    image += new_image
    return image


def find_corners_by_color(prediction, img_size, max_value_y):  # Rename chose_case
    height, width, _ = prediction.shape

    prediction = (prediction * 255).astype(np.uint8)
    prediction = cv2.cvtColor(prediction, cv2.COLOR_RGB2GRAY)
    print(42)
    corner_points = corners_with_color(prediction, img_size)
    print(corner_points)
    if not len(corner_points) == 0:
        floor_corner_count = len(corner_points[corner_points[:, 0] > max_value_y+10])
        ceiling_corner_count = len(corner_points[corner_points[:, 0] < max_value_y+10])
    else:
        floor_corner_count = 0
        ceiling_corner_count = 0
    print(43)
    print(floor_corner_count, ceiling_corner_count)

    if floor_corner_count == 1 and ceiling_corner_count == 0:
        print("one_corner_without_ceiling")
        return one_corner_without_ceiling(prediction, corner_points, height, width, img_size)
    elif floor_corner_count == 1:  # len(corner_points)
        print("one_corner")
        return one_corner_case(prediction, corner_points, height, width, img_size)
    elif floor_corner_count == 2:
        print("two_corner")
        return two_corner_case(prediction, corner_points, height, width)
    elif floor_corner_count == 0:
        print("no_corner")
        return no_corner_case(prediction, height, width, img_size)


def change_color_mean(img, mean, std):
    img = img.astype(np.float64)
    l = img != 0
    orig_mean = np.mean(img[l])
    img[l] = img[l] - orig_mean + mean
    # img[l] = img[l] * (std / np.std(img[l]))
    img = np.clip(img, 0, 255)

    img = img.astype(np.uint8)
    return img


def same_color(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.waitKey(0)
    print(gray)
    n, bins, patches = plt.hist(gray.flatten()[gray.flatten() != 0], bins=50)
    first_max = np.argmax(n)
    max_value_1 = bins[first_max]
    n[first_max] = 0
    second_max = np.argmax(n)
    max_value_2 = bins[second_max]
    new_mean = (max_value_1+max_value_2)/2
    print(n, bins, patches)
    print("new_mean", new_mean)
    print("old_mean", np.mean(gray[gray != 0]))

    th, new_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    cv2.imshow("", new_img)
    cv2.waitKey(0)
    new_img = cv2.connectedComponents(new_img)
    cv2.imshow("", new_img[1].astype(np.uint8))
    cv2.waitKey(0)



def color_matching(walls_mask):
    cv2.imwrite("walls_mask_1616.jpg", walls_mask)
    cv2.imshow("", walls_mask)
    cv2.waitKey(0)
    gray = cv2.cvtColor(walls_mask, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(walls_mask, cv2.COLOR_BGR2HSV)
    only_hue = hsv[..., 0]
    plt.hist(only_hue[only_hue != 0].flatten(), bins=20)
    plt.show()
    cv2.imshow("", only_hue)
    cv2.waitKey(0)
    cv2.imshow("", hsv[..., 1])
    cv2.waitKey(0)
    cv2.imwrite("1616_sat.jpg", hsv[..., 1])
    new_color_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow("", new_color_img.astype(np.uint8))
    cv2.waitKey(0)



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


    print(full_img)
    cv2.imshow("", full_img)
    cv2.waitKey(0)
    cv2.imwrite("full_img.jpg", full_img)
    new_img = (img != 0) * 1

    cv2.waitKey(0)

    color_mask = new_img*np.full(img.shape, [230, 196, 78])

    color_mask_hsv = cv2.cvtColor(color_mask.astype(np.uint8), cv2.COLOR_RGB2HSV)
    color_mask_hsv = color_mask_hsv.astype(np.float64)
    full_img = full_img[..., np.newaxis]

    # change.level(black=-0.5, white=1, channel='all_channels')
    color_mask_hsv[..., 2] = (full_img[..., 0] * color_mask_hsv[..., 2] / 255)
    # new_img_1[..., 1] = (full_img[..., 0] * new_img_1[..., 1] / 255)
    color_mask_hsv[..., 2] *= 1.5
    color_mask_hsv = np.clip(color_mask_hsv, 0, 255)

    cv2.imshow("", color_mask_hsv.astype(np.uint8))
    cv2.waitKey(0)

    color_mask_with_shadows = cv2.cvtColor(color_mask_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    cv2.imshow("", color_mask_with_shadows.astype(np.uint8))
    cv2.waitKey(0)

    hsv = hsv[..., np.newaxis]
    cv2.imwrite("final_result_4.jpg", color_mask_with_shadows)
    cv2.imwrite("without_algo.jpg", (hsv*color_mask_hsv/255).astype(np.uint8))

    return new_img


def main(img_path, out, wood_path, walls_color, segment):

    weights = cwd + "weights/model_retrained.ckpt"
    path = os.path.abspath(img_path)#'/home/vigen/Desktop/perspective_testing/111.jpg'
    warp_img_path = os.path.abspath(wood_path)#'/home/vigen/Desktop/wood4warping/wood1.png'
    img_size = 320
    print(1)
    path_segment = cwd + "mmsegmentation/"
    config_file = path_segment + 'twins_svt-l_uperhead_8x2_512x512_160k_ade20k.py'
    checkpoint_file = path_segment + 'twins_svt-l_uperhead_8x2_512x512_160k_ade20k_20211130_141005-3e2cae61.pth'

    # Read input image
    img = cv2.imread(path)
    orig_height, orig_width = img.shape[:2]
    orig_ratio = orig_height / orig_width
    if img.shape[1] > 2000:
        ratio = img.shape[0] / img.shape[1]
        # cv2.imwrite(img_path.split(".")[0] + "_original.jpg", img)
        img = cv2.resize(img, (2000, int(2000 * ratio)))
        cv2.imwrite(img_path, img)

    height, width = img.shape[:2]
    # Segment input image
    # new_path = cwd[:-1] + "\\" + (img_path.split("/")[-1].split("."))[0]
    # print(cwd)
    # print(new_path)
    print(2)
    print(segment)
    if segment is not None:
        segment = cv2.imread(segment)
        result = segment
        result = result[..., 0]
    else:
        model = init_segmentor(config_file, checkpoint_file, device="cpu")
        result = inference_segmentor(model, path)[0]
        # cv2.imwrite(out + "_mask.png", result)
    result_mask_wall = result == 0
    # noise = np.random.normal(4, 2, height * width * 3).reshape(height, width, 3)
    # noise = np.array([result_mask_wall * noise[..., img_channel] for img_channel in range(3)])
    # noise = np.transpose(noise, (1, 2, 0))

    result_mask_3d = np.array([result_mask_wall for _ in range(3)])
    result_mask_3d = np.transpose(result_mask_3d, (1, 2, 0))

    only_wall_orig = img * result_mask_3d
    # cv2.imwrite(out+"_grayscale.png", cv2.cvtColor(only_wall_orig, cv2.COLOR_BGR2GRAY))

    try:
        if not wood_path is None:
            warp_img = cv2.imread(warp_img_path)
            print(3)
            warp_img = warp_img[:3000, :3000]
            # warp_img = np.transpose(warp_img, (1, 0, 2))
            warp_width = warp_img.shape[1]
            warp_height = warp_img.shape[0]
            ceiling = result == 5

            ceiling1 = ceiling.astype(np.uint8)
            ceiling1 *= 255
            kernel = np.ones((3, 3), np.uint8)
            mask2 = cv2.morphologyEx(ceiling1, cv2.MORPH_OPEN, kernel, iterations=1)
            mask2 = cv2.morphologyEx(255 - mask2, cv2.MORPH_OPEN, kernel, iterations=1)
            mask2 = 255 - mask2
            # cv2.imwrite("mask.jpg", mask2)
            # cv2.imwrite("ceiling.jpg", ceiling1)


            new_array = np.sum(mask2, axis=1)
            max_value_y = round(len(new_array[new_array > 0])/height * img_size) + 4
            # Predict room layout
            prediction = main1(path, weights, img_size)  # Rename main function

            prediction1 = (prediction*255).astype(np.uint8)
            print(41)
            corner_points, corner_old, vanishing_points, corner_count = find_corners_by_color(prediction, img_size, max_value_y)
            vanishing_points = np.array(vanishing_points)
            print(4)

            corner_old = corner_old.astype(np.float64)
            corner_points[:, 0] *= width / img_size
            corner_points[:, 1] *= height / img_size
            # corner_old[:, 0] *= width / img_size
            # corner_old[:, 1] *= height / img_size
            vanishing_points[:, 0] *= width / img_size
            vanishing_points[:, 1] *= height / img_size
            print(vanishing_points)
            corner_points = corner_points.astype(np.int64)
            corner_old = corner_old.astype(np.int64)

            point2 = img
            # Warp image
            pts1 = np.float32([[0, 0], [warp_width, 0], [0, warp_height], [warp_width, warp_height]])
            pts2 = np.float32(corner_points)
            print(5)
            result_mask_floor = result == 3


            if corner_count == "two_corner":
                print("two_corner")
                pts2 = black_cases_2_corner(vanishing_points[0], vanishing_points[1], result_mask_floor)
            elif corner_count == "one_corner":
                print("one_corner")
                pts2 = black_cases_1_corner(vanishing_points[0], vanishing_points[1], result_mask_floor)
            elif corner_count == "one_corner_case_without_ceiling":
                print("one_corner_case_without_ceiling")
                pts2 = black_cases_1_corner(vanishing_points[0], vanishing_points[1], result_mask_floor)
                print(pts2)
            elif corner_count == "no_corner":
                print("no_corner")
                pts2 = black_cases_1_corner(vanishing_points[0], vanishing_points[1], result_mask_floor)
            elif corner_count == "no_corner_vertical":
                print("no_corner_vertical")
                pts2 = no_corner_vertical(result_mask_floor)



            matrix_new = cv2.getPerspectiveTransform(pts1, np.float32(np.array(pts2)))
            img_output = cv2.warpPerspective(warp_img, matrix_new, (width, height))

            # new_color = np.full((300, 300, 3), [100, 20, 30])
            # new_color[:100] = np.full((100, 300, 3), [0, 0, 0])
            # cv2.imshow("", new_color.astype(np.uint8))
            # cv2.waitKey(0)
            # unique_color = np.unique(np.unique(new_color, axis=0), axis=1)[:, 0]
            # print(unique_color)
            # is_same = False
            # try:
            #     color = unique_color[1]
            #     print((img_output != [0, 0, 0]))
            #     cv2.imshow("", (np.all((img_output == color), axis=2) * 30).astype(np.uint8))
            #     cv2.waitKey(0)
            #     cv2.imshow("", (np.all((img_output != [0, 0, 0]), axis=2) * 30).astype(np.uint8))
            #     cv2.waitKey(0)
            #     if np.all(np.all((img_output == color), axis=2) == np.all((img_output != [0, 0, 0]), axis=2)):
            #         is_same = True
            # except:
            #     color = unique_color[0]
            #     if np.all(np.all((img_output == color), axis=2) == np.all((img_output != [0, 0, 0]), axis=2)):
            #         is_same = True
            # print(is_same)
            # assert 0
            if np.sum((~(img_output != 0))[..., 0]*result_mask_floor) > 100:
                print(np.sum((~img_output != 0)[..., 0]*result_mask_floor))
                print("default_perspective")
                pts2 = default_perspective(result_mask_floor)
                matrix_new = cv2.getPerspectiveTransform(pts1, np.float32(np.array(pts2)))
                img_output = cv2.warpPerspective(warp_img, matrix_new, (width, height))


            # cv2.imshow("", img_output)
            # cv2.waitKey(0)


            # result_mask_floor = cv2.resize((result_mask_floor*1), (682, 1024), interpolation=cv2.INTER_AREA)

            img_output = np.array([img_output[:, :, img_channel] * result_mask_floor for img_channel in range(3)])
            img_output = img_output.transpose([1, 2, 0])

            floor_mask_3d = np.array([result_mask_floor for _ in range(3)])
            floor_mask_3d = np.transpose(floor_mask_3d, (1, 2, 0))
            original_floor_mask = img * floor_mask_3d

            # Test

            wand_floor_2change = img_output #transport_shadows(original_floor_mask, img_output) # img_output

            img = mask2image(result_mask_floor, img, wand_floor_2change)
        # Wall adding and noising new color
        if not walls_color is None:
            result_mask_wall = result == 0
            # noise = np.random.normal(4, 2, height * width * 3).reshape(height, width, 3)
            # noise = np.array([result_mask_wall * noise[..., img_channel] for img_channel in range(3)])
            # noise = np.transpose(noise, (1, 2, 0))

            result_mask_3d = np.array([result_mask_wall for _ in range(3)])
            result_mask_3d = np.transpose(result_mask_3d, (1, 2, 0))

            only_wall_orig = img * result_mask_3d
            # cv2.imwrite("wall_grayscale.jpg", cv2.cvtColor(only_wall_orig, cv2.COLOR_BGR2GRAY))
            walls_color = walls_color[::-1]

            img[result_mask_wall] = np.array(walls_color)
            img = img.astype(np.float32)
            #img += noise
            img = img.astype(np.uint8)
            only_wall = img * result_mask_3d
            # cv2.imshow("", only_wall)
            # cv2.waitKey(0)

            # Wand wall color
            # same_color(only_wall_orig)
            # wand_wall_2change = transport_shadows(only_wall_orig, only_wall)
            # color_matching(only_wall_orig)
            # wall_same_light(only_wall_orig, corner_old)



            wand_wall_2change = transport_shadows(only_wall_orig, only_wall)

            img = mask2image(result_mask_wall, img, wand_wall_2change)

        # Wand floor color


        out = out + ".jpg"

        # floor_rgba = cv2.imread(cwd + "logos/Floor.png", -1)
        # wall_rgba = cv2.imread(cwd + "logos/Overlay.png", -1)

        #img = add_logo(img, result_mask_floor, result_mask_wall, floor_rgba, wall_rgba)

        cv2.imwrite(out, img)
    except Exception as e:
        print("Caught an Exception", e)
        #mark_an_image(img_path, out)
        weights = cwd + "weights/model_retrained.ckpt"

        result_mask_floor = result == 3
        warp_img = cv2.imread(warp_img_path)
        print(3)
        warp_img = warp_img[:3000, :3000]
        # warp_img = np.transpose(warp_img, (1, 0, 2))
        warp_width = warp_img.shape[1]
        warp_height = warp_img.shape[0]
        pts1 = np.float32([[0, 0], [warp_width, 0], [0, warp_height], [warp_width, warp_height]])
        pts2 = default_perspective(result_mask_floor)
        print(pts2)
        matrix_new = cv2.getPerspectiveTransform(pts1, np.float32(np.array(pts2)))
        img_output = cv2.warpPerspective(warp_img, matrix_new, (width, height))

        img_output = np.array([img_output[:, :, img_channel] * result_mask_floor for img_channel in range(3)])
        img_output = img_output.transpose([1, 2, 0])

        floor_mask_3d = np.array([result_mask_floor for _ in range(3)])
        floor_mask_3d = np.transpose(floor_mask_3d, (1, 2, 0))
        original_floor_mask = img * floor_mask_3d

        # Test

        wand_floor_2change = img_output  # transport_shadows(original_floor_mask, img_output) # img_output

        img = mask2image(result_mask_floor, img, wand_floor_2change)

        out = out + ".jpg"

        cv2.imwrite(out, img)

def analyse_input_image(path):
    import json
    from sample_images_urls import sample_images_urls

    output = sample_images_urls.get(path)
    if output is not None:
        walls_color, tile_name, _ = output
        tile_name = "wood/" + tile_name + ".png"
    else:
        command = f"cd /root && python3 floor_swap/image_analyzer_wraper.py --img {path}"
        os.system(command)

        f = open(path[:-3] + "json")
        data = json.load(f)["look_info"]
        tile_name = data["floor_tile"]
        walls_color = data["walls_color"]
        walls_color = list(map(int, walls_color.split(" ")))

    tile_path = tiles_folder + tile_name


    return tile_path, walls_color


def mark_an_image(img_path, out):
    im = cv2.imread(img_path)
    h, w, _ = im.shape

    sample_text = cv2.imread(cwd + "sample_text.png")
    sample_text = cv2.resize(sample_text, (w, h))

    im = im*0.5
    res = np.where(sample_text == 255, sample_text, im).astype("uint8")
    out = out + ".jpg"
    cv2.imwrite(out, res)


if __name__ == '__main__':
    import time

    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, required=True)
    parser.add_argument('--replace_walls', nargs="+", type=int)
    parser.add_argument('--replace_floor', type=str)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--mask', type=str)
    args = parser.parse_args()

    img_path = args.img
    out = args.out
    add_walls = args.replace_walls
    wood_path = args.replace_floor
    segment = args.mask
    # wood_path, walls_color = analyse_input_image(args.analyse)

    if not add_walls is None:
        walls_color = np.array(add_walls, dtype=np.uint8)
        color1 = walls_color.tolist()
        wall_colors = np.array(color1 * 300 * 300).reshape(300, 300, 3)

        # cv2.imwrite("color.png", wall_colors)
    else:
        walls_color = add_walls
    start = time.time()

    main(img_path, out, wood_path, walls_color, segment)
    end = time.time()
    print(end-start)
    # try:
    #     main(img_path, out, wood_path, walls_color)
    # except Exception as e:
    #     print("Caught an Exception", e)
    #     #mark_an_image(img_path, out)
    #     weights = cwd + "weights/model_retrained.ckpt"
    #     path = os.path.abspath(img_path)  # '/home/vigen/Desktop/perspective_testing/111.jpg'
    #     warp_img_path = os.path.abspath(wood_path)  # '/home/vigen/Desktop/wood4warping/wood1.png'
    #     img_size = 320
    #     print(1)
    #     path_segment = cwd + "mmsegmentation/"
    #     config_file = path_segment + 'twins_svt-l_uperhead_8x2_512x512_160k_ade20k.py'
    #     checkpoint_file = path_segment + 'twins_svt-l_uperhead_8x2_512x512_160k_ade20k_20211130_141005-3e2cae61.pth'
    #
    #     # Read input image
    #     img = cv2.imread(path)
    #     orig_height, orig_width = img.shape[:2]
    #     orig_ratio = orig_height / orig_width
    #     if orig_width >= 2000:
    #         img = cv2.resize(img, (2000, int(orig_ratio * 2000)), cv2.INTER_LANCZOS4)
    #
    #     height, width = img.shape[:2]
    #     # Segment input image
    #     model = init_segmentor(config_file, checkpoint_file, device='cuda')
    #     print(2)
    #
    #     result = inference_segmentor(model, path)[0]
    #     result_mask_floor = result == 3
    #     warp_img = cv2.imread(warp_img_path)
    #     print(3)
    #     warp_img = warp_img[:3000, :3000]
    #     # warp_img = np.transpose(warp_img, (1, 0, 2))
    #     warp_width = warp_img.shape[1]
    #     warp_height = warp_img.shape[0]
    #     pts1 = np.float32([[0, 0], [warp_width, 0], [0, warp_height], [warp_width, warp_height]])
    #     pts2 = default_perspective(result_mask_floor)
    #     matrix_new = cv2.getPerspectiveTransform(pts1, np.float32(np.array(pts2)))
    #     img_output = cv2.warpPerspective(warp_img, matrix_new, (width, height))
    #
    #     img_output = np.array([img_output[:, :, img_channel] * result_mask_floor for img_channel in range(3)])
    #     img_output = img_output.transpose([1, 2, 0])
    #
    #     floor_mask_3d = np.array([result_mask_floor for _ in range(3)])
    #     floor_mask_3d = np.transpose(floor_mask_3d, (1, 2, 0))
    #     original_floor_mask = img * floor_mask_3d
    #
    #     # Test
    #
    #     wand_floor_2change = img_output  # transport_shadows(original_floor_mask, img_output) # img_output
    #
    #     img = mask2image(result_mask_floor, img, wand_floor_2change)
    #
    #     out = out + ".jpg"
    #
    #     cv2.imwrite(out, img)
    # main(img_path, out, wood_path, walls_color)
    # exe_time = time.time() - start
    # print("Processing time:", np.round(exe_time, 2))
