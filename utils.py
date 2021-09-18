import numpy as np
import cv2
import matplotlib.pyplot as plt


# Normalize image
def normalize(image, img_type='gray'):
    if img_type == 'gray':
        image = image / 127.5 - 1
    else:
        image = image / 255

    return image


# Denormarlize image
def denormalize(image, img_type='gray'):
    if img_type == 'gray':
        image = (image + 1) * 127.5
    else:
        image = image * 255
    return image


def remove_small_blobs(bw_image, min_size=3000):

    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bw_image, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    image = np.zeros((bw_image.shape))
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            image[output == i + 1] = 255

    return image


def find_KUB_bounding_box(full_KUB_map, border_size=0.1):

    # KUB map preprocessing
    full_KUB_map = np.array(full_KUB_map, dtype=np.uint8)
    # _, full_KUB_map = cv2.threshold(full_KUB_map, 0, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    full_KUB_map = cv2.dilate(full_KUB_map, kernel, iterations=2)

    # find contour
    cnt_tmp = cv2.findContours(full_KUB_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = cnt_tmp[0] if len(cnt_tmp) == 2 else cnt_tmp[1]
    x, y, w, h = cv2.boundingRect(contour[0])    # parameters for L and R partitions

    border_w = 30 #int(border_size * w)
    border_h = 30 #int(border_size * h)

    # expand bb
    x_top = max(0, x - border_w)
    w_top = min(w + 2*border_w, full_KUB_map.shape[1] - x_top)
    y_top = max(0, y - border_h)
    h_top = round(h/2) + 2*border_h

    # find bladder partition
    low_img_map = full_KUB_map.copy()
    low_img_map[y:y+round(h/2), :] = 0    # remove top map

    # find contour
    cnt_tmp = cv2.findContours(low_img_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    low_contour = cnt_tmp[0] if len(cnt_tmp) == 2 else cnt_tmp[1]
    x_low, y_low, w_low, h_low = cv2.boundingRect(low_contour[0])  # parameters for B partitio

    # expand bb
    x_low = x_low - border_w
    w_low = w_low + 2 * border_w
    y_low = y_low - border_h
    h_low = min(h_low + 2*border_h, full_KUB_map.shape[0] - y_low)

    # # display testing
    # display_img = np.zeros([full_KUB_map.shape[0], full_KUB_map.shape[1], 3])
    # display_img[:, :, 0] = full_KUB_map.copy()
    # display_img[:, :, 1] = full_KUB_map.copy()
    # display_img[:, :, 2] = full_KUB_map.copy()
    #
    # # left partition
    # cv2.rectangle(display_img, (x_top, y_top), (x_top + round(w_top/2), y_top + h_top), (255, 0, 0), 3)
    # # right partition
    # cv2.rectangle(display_img, (1 + x_top + round(w_top/2), y_top), (x_top + w_top, y_top + h_top), (0, 255, 0), 3)
    # # bottom partition
    # cv2.rectangle(display_img, (x_low, y_low), (x_low + w_low, y_low + h_low), (0, 0, 255), 3)
    #
    # cv2.imshow('bb_display', display_img)
    # cv2.waitKey(0)

    return x_top, y_top, w_top, h_top, x_low, y_low, w_low, h_low


def create_KUB_partitions(full_image, full_KUB_map):

    # bb of full KUB map
    x_top, y_top, w_top, h_top, x_low, y_low, w_low, h_low = find_KUB_bounding_box(full_KUB_map)

    left_partition = full_image[y_top:y_top+h_top, x_top:x_top+round(w_top/2)]
    right_partition = full_image[y_top:y_top+h_top, x_top+round(w_top/2):x_top+w_top]
    bottom_partition = full_image[y_low:y_low+h_low, x_low:x_low+w_low]

    return left_partition, right_partition, bottom_partition


def combine_KUB_partitions(left_image, right_image, bladder_image, full_KUB_map):

    # bb of full KUB map
    x_top, y_top, w_top, h_top, x_low, y_low, w_low, h_low = find_KUB_bounding_box(full_KUB_map)

    # resize input partitions
    left_image = cv2.resize(left_image, (round(w_top/2), h_top))
    right_image = cv2.resize(right_image, (w_top - round(w_top/2), h_top))
    bottom_image = cv2.resize(bladder_image, (w_low, h_low))

    # create image size equal size to full image
    combined_image = np.zeros([full_KUB_map.shape[0], full_KUB_map.shape[1]], np.float32)

    combined_image[y_top:y_top+h_top, x_top:x_top+round(w_top/2)] = left_image            # left partition
    combined_image[y_top:y_top+h_top, x_top+round(w_top/2):x_top+w_top] = right_image      # right partition
    combined_image[y_low:y_low+h_low, x_low:x_low+w_low] = combined_image[y_low:y_low + h_low, x_low:x_low + w_low] \
                                                           + bottom_image   # bottom partition

    # cv2.imshow('combined_image', combined_image)
    # cv2.waitKey(0)

    return combined_image


def create_KUB_partitions_old(full_image):

    w, h = full_image.shape

    left_partition = full_image[0:round(h/2), 0:round(w/2)]
    right_partition = full_image[0:round(h/2), 1+round(w/2):w]
    bladder_partition = full_image[1 + round(h/2):h, round(w/4):3 * round(w/4)]

    return left_partition, right_partition, bladder_partition


def make_square(img):
    s_size = max(img.shape[0:2])
    new_image = np.zeros((s_size, s_size), np.float32)

    # Getting the centering position
    ax, ay = (s_size - img.shape[1]) // 2, (s_size - img.shape[0]) // 2

    # Pasting the 'image' in a centering position
    new_image[ay:img.shape[0] + ay, ax:ax + img.shape[1]] = img

    return new_image


def heatmap_generate(org_image, predicted):

    heatmap = cv2.resize(predicted, (org_image.shape[1], org_image.shape[0]))
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    hif = .8
    superimposed_img = heatmap * hif + org_image

    return superimposed_img
