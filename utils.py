import numpy as np
import cv2


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


def remove_small_blobs(bw_image, min_size=1000):
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bw_image, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    image = np.zeros((output.shape))
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            image[output == i + 1] = 255

    return image


def find_KUB_bounding_box(full_KUB_map, border_size=30):

    # KUB map preprocessing
    full_KUB_map = np.array(full_KUB_map, dtype=np.uint8)
    _, full_KUB_map = cv2.threshold(full_KUB_map, 0, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    full_KUB_map = cv2.dilate(full_KUB_map, kernel, iterations=1)

    # find contour
    cnt_tmp = cv2.findContours(full_KUB_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = cnt_tmp[0] if len(cnt_tmp) == 2 else cnt_tmp[1]
    x, y, w, h = cv2.boundingRect(contour[0])    # parameters for L and R partitions

    # expand bb
    x_top = max(0, x - border_size)
    w_top = min(w + 2*border_size, full_KUB_map.shape[1] - x_top)
    y_top = max(0, y - border_size)
    h_top = round(h/2) + 2*border_size

    # find bladder partition
    low_img_map = full_KUB_map.copy()
    low_img_map[y:y+round(h/2), :] = 0    # remove top map

    # find contour
    cnt_tmp = cv2.findContours(low_img_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    low_contour = cnt_tmp[0] if len(cnt_tmp) == 2 else cnt_tmp[1]
    x_low, y_low, w_low, h_low = cv2.boundingRect(low_contour[0])  # parameters for B partition
    # expand bb
    y_low = y_low - border_size
    x_low = x_low - border_size
    w_low = w_low + 2 * border_size
    h_low = min(h_low + 2*border_size, full_KUB_map.shape[0] - y_low)

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


