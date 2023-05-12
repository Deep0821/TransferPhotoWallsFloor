import cv2
import numpy as np
import torch
from datasets import sequence
from trainer import core
from mmseg.apis import inference_segmentor, init_segmentor
from wand.image import Image
import argparse

from wand.display import display
import os

cwd = os.path.dirname(os.path.abspath(__file__)) + '/'
tiles_folder = "/root/floor_swap/floor_module/floor_material/"
# torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()


class Predictor:
    def __init__(self, weight_path):
        self.model = core.LayoutSeg.load_from_checkpoint(weight_path, backbone='resnet101')
        self.model.freeze()
        self.model.eval()

    @torch.no_grad()
    def feed(self, image: torch.Tensor) -> np.ndarray:
        _, outputs = self.model(image[None, :, :, :].cpu())
        label = core.label_as_rgb_visual(outputs.cpu()).squeeze(0)
        return label.permute(1, 2, 0).numpy()


def isnt_duplicate(check_point, anchor_points, criterion=10):
    check_point = np.array(check_point)
    for anchor_point in anchor_points:
        anchor_point = np.array(anchor_point)
        if np.all((check_point < anchor_point + criterion) & (check_point > anchor_point - criterion)):
            return False
            break
    else:
        return True


def main1(path, weight, image_size):  # Predict
    predictor = Predictor(weight_path=weight)
    image = sequence.ImageFolder(image_size, path)
    image = image.test()
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
    # print(corner_points)
    return np.array(corner_points)


def transport_shadows(img1, img2):
    img1 = Image.from_array(img1)
    img2 = Image.from_array(img2)
    img1.transform_colorspace('gray')

    img1.level(black=-0.5, white=1, channel='all_channels')
    # img_new = np.array(img1)
    # cv2.imwrite("shadows2.jpg", img_new[..., 0])
    # new_img = cv2.GaussianBlur(img_new[..., 0], (71, 71), 0)
    # new_img = Image.from_array(new_img)
    # cv2.imwrite("blured.jpg",  cv2.GaussianBlur(img_new[..., 0], (71, 71), 0))
    # cv2.imwrite("wall_shadows.jpg", img1)
    img2.composite_channel('all_channels', img1, 'hard_light', 0, 0)
    img2 = np.array(img2)
    return img2


def mask2image(mask, image_orig, image_transform):
    image_orig[mask] = [0, 0, 0]
    image_orig += image_transform
    image_orig.astype(np.uint8)
    return image_orig


def one_corner_case(prediction, corner_points, height, width, img_size):
    # print("one corner")
    edges = cv2.Canny(prediction, 20, 200)
    edges_upper = edges.copy()
    edges_bottom = edges.copy()

    edges_upper[height // 2:] = 0
    edges_bottom[:height // 2] = 0

    corner_points = corner_points[:, ::-1]
    have_upper = np.array([np.sum(edges_upper[:, column]) for column in range(width)])
    have_bottom = np.array([np.sum(edges_bottom[:, column]) for column in range(width)])
    left_upper_line_x, right_upper_line_x = np.argwhere(have_upper != 0)[[0, -1]]
    left_line_x, right_line_x = np.argwhere(have_bottom != 0)[[0, -1]]

    left_max = [left_upper_line_x[0], np.argwhere(edges[:, left_upper_line_x] != 0)[0, 0]]
    right_max = [right_upper_line_x[0], np.argwhere(edges[:, right_upper_line_x] != 0)[0, 0]]

    right_upper_line_k, right_upper_line_b = find_k_and_b(corner_points[0], right_max)

    left_min = [left_line_x[0], np.argwhere(edges_bottom[:, left_line_x] != 0)[-1, 0]]
    right_min = [right_line_x[0], np.argwhere(edges_bottom[:, right_line_x] != 0)[-1, 0]]

    upper_line_k, upper_line_b = find_k_and_b(left_max, corner_points[0])
    corner_points = corner_points[-1]
    left_line_k, left_line_b = find_k_and_b(corner_points, left_min)
    right_line_k, right_line_b = find_k_and_b(corner_points, right_min)

    right_line_inter_x, right_line_inter_y = two_line_intersection([right_line_k, right_line_b],
                                                                   [right_upper_line_k, right_upper_line_b])
    new_line_k, new_line_b = find_k_and_b([right_line_inter_x, right_line_inter_y], left_min)
    point_2 = left_min
    point_3 = [(img_size - new_line_b) / new_line_k, img_size]
    point_4 = [(img_size - right_line_b) / right_line_k, img_size]
    new_and_right_x, new_and_right_y = two_line_intersection([new_line_k, new_line_b], [right_line_k, right_line_b])
    left_and_vanishing_x, left_and_vanishing_y = (new_and_right_y - left_line_b) / left_line_k, new_and_right_y

    right_new_line_k, right_new_line_b = find_k_and_b([left_and_vanishing_x, left_and_vanishing_y], point_4)

    need_point = two_line_intersection([right_new_line_k, right_new_line_b], [new_line_k, new_line_b])

    corner_new_points = [left_min]
    corner_new_points.append(corner_points)
    corner_new_points.append(need_point)
    corner_new_points.append(point_4)

    corner_new_points = np.array(corner_new_points)
    return corner_new_points


def two_corner_case(prediction, corner_points, height, width):
    # Find image-border points
    # print("two_corner")
    edges = cv2.Canny(prediction, 20, 200)
    edges[:height // 2] = 0
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

    # Find slope and bias for 3 lines
    left_line_k, left_line_b = find_k_and_b(corner_points[0], left_min)
    right_line_k, right_line_b = find_k_and_b(corner_points[1], right_min)
    upper_line_k, upper_line_b = find_k_and_b(corner_points[0], corner_points[1])

    # Find outer scope points
    x1 = -left_line_b / left_line_k
    x2 = -right_line_b / right_line_k

    # Determine room case
    x_choose = x2 if corner_points[0][1] < corner_points[1][1] else x1
    lines_choose_k, lines_choose_b = (left_line_k, left_line_b)
    if corner_points[0][1] > corner_points[1][1]:
        lines_choose_k, lines_choose_b = (right_line_k, right_line_b)

    # Find vanishing point
    vanishing_point_x, vanishing_point_y = two_line_intersection([left_line_k, left_line_b],
                                                                 [right_line_k, right_line_b])
    upper_line_vanishing_point = (vanishing_point_y - upper_line_b) / upper_line_k

    bottom_line_k, bottom_line_b = find_k_and_b([upper_line_vanishing_point, vanishing_point_y], [x_choose, 0])
    # Use two_line_intersection function
    x_intersection = (bottom_line_b - lines_choose_b) / (lines_choose_k - bottom_line_k)
    y_intersection = bottom_line_k * x_intersection + bottom_line_b

    if x_choose == x1:
        corner_points = np.append(corner_points, [[x_choose, 0]], axis=0)
        corner_points = np.append(corner_points, [[x_intersection, y_intersection]], axis=0)
    else:

        corner_points = np.append(corner_points, [[x_intersection, y_intersection]], axis=0)
        corner_points = np.append(corner_points, [[x_choose, 0]], axis=0)

    return corner_points


def no_corner_case(prediction, height, width, img_size):
    edges = cv2.Canny(prediction, 20, 200)
    edges_upper = edges.copy()
    edges_bottom = edges.copy()
    edges_bottom[:height // 2] = 0
    edges_upper[height // 2:] = 0

    have_upper = np.array([np.sum(edges_upper[:, column]) for column in range(width)])
    have_bottom = np.array([np.sum(edges_bottom[:, column]) for column in range(width)])
    left_upper_line_x, right_upper_line_x = np.argwhere(have_upper != 0)[[0, -1]]
    left_line_x, right_line_x = np.argwhere(have_bottom != 0)[[0, -1]]

    left_max = [left_upper_line_x[0], np.argwhere(edges[:, left_upper_line_x] != 0)[0, 0]]
    right_max = [right_upper_line_x[0], np.argwhere(edges[:, right_upper_line_x] != 0)[0, 0]]

    left_min = [left_line_x[0], np.argwhere(edges_bottom[:, left_line_x] != 0)[-1, 0]]
    right_min = [right_line_x[0], np.argwhere(edges_bottom[:, right_line_x] != 0)[-1, 0]]

    line_k, line_b = find_k_and_b(left_min, right_min)

    if np.arctan(line_k) < 4 * np.pi / 180 and np.arctan(line_k) > -4 * np.pi / 180:
        central_point = [round(width / 2), round(height / 2)]
        new_right_k, new_right_b = find_k_and_b(central_point, right_min)
        new_left_k, new_left_b = find_k_and_b(central_point, left_min)
        x_left = (height - new_left_b) / new_left_k
        x_right = (height - new_right_b) / new_right_k
        corner_points = [left_min, right_min, [x_left, height], [x_right, height]]
        return np.array(corner_points)

    upper_line_k, upper_line_b = find_k_and_b(left_max, right_max)
    inter_x, inter_y = two_line_intersection((line_k, line_b), (upper_line_k, upper_line_b))

    if left_min[1] < right_min[1]:

        bottom_line_k, bottom_line_b = find_k_and_b([0, height], [inter_x, inter_y])
        vanishing_point = [width - round(width / 4), inter_y]
        left_new_line_k, left_new_line_b = find_k_and_b(vanishing_point, left_min)
        right_new_line_k, right_new_line_b = find_k_and_b(vanishing_point, right_min)
        bottom_left_inter_x, bottom_left_inter_y = two_line_intersection((left_new_line_k, left_new_line_b),
                                                                         (bottom_line_k, bottom_line_b))
        bottom_right_inter_x, bottom_right_inter_y = two_line_intersection((right_new_line_k, right_new_line_b),
                                                                           (bottom_line_k, bottom_line_b))

        corner_points = [left_min, (bottom_left_inter_x, bottom_left_inter_y), right_min,
                         (bottom_right_inter_x, bottom_right_inter_y)]

    elif left_min[1] > right_min[1]:

        bottom_line_k, bottom_line_b = find_k_and_b([width, height], [inter_x, inter_y])
        vanishing_point = [round(width / 4), inter_y]
        right_new_line_k, right_new_line_b = find_k_and_b(vanishing_point, right_min)
        left_new_line_k, left_new_line_b = find_k_and_b(vanishing_point, left_min)
        bottom_right_inter_x, bottom_right_inter_y = two_line_intersection((right_new_line_k, right_new_line_b),
                                                                           (bottom_line_k, bottom_line_b))
        bottom_left_inter_x, bottom_left_inter_y = two_line_intersection((left_new_line_k, left_new_line_b),
                                                                         (bottom_line_k, bottom_line_b))
        corner_points = [(bottom_right_inter_x, bottom_right_inter_y), right_min,
                         (bottom_left_inter_x, bottom_left_inter_y), left_min]

    return np.array(corner_points)


def lightest_pixel(distance, img_shape):
    distance_height, distance_width = distance.shape
    distance = np.array(distance)
    # cv2.imwrite("img1.png", distance)
    brightness = [0] * img_shape[1]
    rows = [0] * img_shape[1]
    for column in range(img_shape[1], distance_width - img_shape[1]):
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

    floor_mask = floor_mask * 255
    wall_mask = wall_mask * 255

    image_size_floor = 100
    image_size_wall = 70
    floor_mask = floor_mask.astype(np.uint8)
    wall_mask = wall_mask.astype(np.uint8)
    floor_distance = cv2.distanceTransform(floor_mask, cv2.DIST_L2, 3)
    wall_distance = cv2.distanceTransform(wall_mask, cv2.DIST_L2, 3)
    # cv2.imwrite("img1.png", floor_distance)
    # cv2.imwrite("img2.png", wall_distance)
    image_ratio_floor = floor_image.shape[0] / floor_image.shape[1]
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
    corner_points = corners_with_color(prediction, img_size)

    floor_corner_count = len(corner_points[corner_points[:, 0] > max_value_y])
    # print(max_value_y)
    # print(floor_corner_count)

    if floor_corner_count == 1:  # len(corner_points)
        print("one_corner")
        return one_corner_case(prediction, corner_points, height, width, img_size)
    elif floor_corner_count == 2:
        print("two_corner")
        return two_corner_case(prediction, corner_points, height, width)
    elif floor_corner_count == 0:
        print("no_corner")
        return no_corner_case(prediction, height, width, img_size)


def main(img_path, out, wood_path, walls_color):
    weights = cwd + "weights/model_retrained.ckpt"
    path = os.path.abspath(img_path)  # '/home/vigen/Desktop/perspective_testing/111.jpg'

    img_size = 320

    path_segment = cwd + "mmsegmentation/"
    config_file = path_segment + 'twins_svt-l_uperhead_8x2_512x512_160k_ade20k.py'
    checkpoint_file = path_segment + 'twins_svt-l_uperhead_8x2_512x512_160k_ade20k_20211130_141005-3e2cae61.pth'

    # Read input image
    img = cv2.imread(path)
    orig_height, orig_width = img.shape[:2]
    orig_ratio = orig_height / orig_width
    img = cv2.resize(img, (1024, int(orig_ratio * 1024)), cv2.INTER_LANCZOS4)

    # cv2.imwrite("test_resize.png", img)
    height, width = img.shape[:2]
    # print(height, width)
    # Segment input image
    model = init_segmentor(config_file, checkpoint_file, device='cpu')
    result = inference_segmentor(model, path)[0]
    result = cv2.resize(result.astype(np.uint8), (width, height))


    # warp_img = np.transpose(warp_img, (1, 0, 2))

    # cv2.imwrite("mask.jpg", mask2)
    # cv2.imwrite("ceiling.jpg", ceiling1)

    # print(ceiling)

    # Predict room layout


    # Wall adding and noising new color
    result_mask_wall = result == 0
    # noise = np.random.normal(4, 2, height * width * 3).reshape(height, width, 3)
    # noise = np.array([result_mask_wall * noise[..., img_channel] for img_channel in range(3)])
    # noise = np.transpose(noise, (1, 2, 0))
    if not walls_color is None:
        result_mask_3d = np.array([result_mask_wall for _ in range(3)])
        result_mask_3d = np.transpose(result_mask_3d, (1, 2, 0))

        only_wall_orig = img * result_mask_3d

        img[result_mask_wall] = np.array(walls_color)

        img = img.astype(np.float32)
        # img += noise
        img = img.astype(np.uint8)
        only_wall = img * result_mask_3d

        #Wand wall color
        hsv = cv2.cvtColor(only_wall_orig, cv2.COLOR_RGB2HSV)
        mean_v = np.mean(hsv[:, :, 2])

        wand_wall_2change = transport_shadows(only_wall_orig, only_wall)

        img = mask2image(result_mask_wall, img, wand_wall_2change)

    # cv2.imshow("", img)
    # cv2.waitKey(0)

    result_mask_floor = result == 3
    #
    if not wood_path is None:
        warp_img_path = os.path.abspath(wood_path)  # '/home/vigen/Desktop/wood4warping/wood1.png'
        warp_img = cv2.imread(warp_img_path)
        warp_img = warp_img[:1500, :1500]

        warp_width = warp_img.shape[1]
        warp_height = warp_img.shape[0]
        ceiling = result == 5

        ceiling1 = ceiling.astype(np.uint8)
        ceiling1 *= 255
        kernel = np.ones((3, 3), np.uint8)
        mask2 = cv2.morphologyEx(ceiling1, cv2.MORPH_OPEN, kernel, iterations=1)
        mask2 = cv2.morphologyEx(255 - mask2, cv2.MORPH_OPEN, kernel, iterations=1)
        mask2 = 255 - mask2

        new_array = np.sum(mask2, axis=1)
        max_value_y = round(len(new_array[new_array > 0]) / height * img_size) + 4
        prediction = main1(path, weights, img_size)  # Rename main function
        prediction1 = (prediction * 255).astype(np.uint8)

        corner_points = find_corners_by_color(prediction, img_size, max_value_y)
        corner_points[:, 0] *= width / img_size
        corner_points[:, 1] *= height / img_size
        corner_points = corner_points.astype(np.int64)
        point2 = img

        # Warp image
        pts1 = np.float32([[0, 0], [warp_width, 0], [0, warp_height], [warp_width, warp_height]])
        pts2 = np.float32(corner_points)
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_output = cv2.warpPerspective(warp_img, matrix, (width, height))

        # Mask warped tile image


        img_output = np.array([img_output[:, :, img_channel] * result_mask_floor for img_channel in range(3)])
        img_output = img_output.transpose([1, 2, 0])

        # cv2.imshow("", img_output)
        # cv2.waitKey(0)

        # Wand floor color
        floor_mask_3d = np.array([result_mask_floor for _ in range(3)])
        floor_mask_3d = np.transpose(floor_mask_3d, (1, 2, 0))
        original_floor_mask = img * floor_mask_3d

        # Test
        wand_floor_2change = transport_shadows(original_floor_mask, img_output)

        img = mask2image(result_mask_floor, img, wand_floor_2change)

    out = out + ".jpg"

    # floor_rgba = cv2.imread(cwd + "logos/Floor.png", -1)
    # wall_rgba = cv2.imread(cwd + "logos/Overlay.png", -1)
    #
    # img = add_logo(img, result_mask_floor, result_mask_wall, floor_rgba, wall_rgba)

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
    print(tile_path, walls_color)

    return tile_path, walls_color


def mark_an_image(img_path, out):
    im = cv2.imread(img_path)
    h, w, _ = im.shape

    sample_text = cv2.imread(cwd + "sample_text.png")
    sample_text = cv2.resize(sample_text, (w, h))

    im = im * 0.5
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

    args = parser.parse_args()

    img_path = args.img
    out = args.out
    add_walls = args.replace_walls
    wood_path = args.replace_floor

    # wood_path, walls_color = analyse_input_image(args.analyse)

    if not add_walls is None:
        walls_color = np.array(add_walls, dtype=np.uint8)
        color1 = walls_color.tolist()
        wall_colors = np.array(color1 * 300 * 300).reshape(300, 300, 3)

        # cv2.imwrite("color.png", wall_colors)
    else:
        walls_color = add_walls

    try:
        main(img_path, out, wood_path, walls_color)
    except Exception as e:
        print("Caught an Exception", e)
        mark_an_image(img_path, out)

    # main(img_path, out, wood_path, walls_color)
    # exe_time = time.time() - start
    # print("Processing time:", np.round(exe_time, 2))
