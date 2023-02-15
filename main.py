import tensorflow as tf
import numpy as np
import cv2
import os
# from unet_model import UNet
from unet_based_model import UNet
from utils import create_KUB_partitions, combine_KUB_partitions, normalize, \
    denormalize, remove_small_blobs, make_square, heatmap_generate
import matplotlib.pyplot as plt


# define configuration
cfg = {'image_size': 256,
       'full_image_size': 1024,
       'cascade_unet': True}


def load_images_from_folder(folder_name, image_size, resize_method=None, image_type='gray'):
    images_list = os.listdir(folder_name)
    images = np.zeros((len(images_list), image_size, image_size, 1), np.float32)

    for i, image_name in enumerate(images_list):

        # read image
        image = cv2.imread(folder_name + os.sep + image_name, cv2.IMREAD_GRAYSCALE)

        # resize
        if resize_method == 'fill':
            image = make_square(image)
        image = cv2.resize(image, (image_size, image_size))

        image = image[:, :, np.newaxis]

        if image_type == 'bw':
            images[i] = normalize(image, img_type='bw')
        elif image_type == 'gray':
            images[i] = normalize(image, img_type='gray')

    return images


# 1st-stage : KUB map generation
def kub_map_generation(full_images):

    # preprocessing (256 x 256)
    def resize_tensor(tensor):
        new_tensor = np.zeros((tensor.shape[0], cfg['image_size'], cfg['image_size'], 1), np.float32)
        for i in range(tensor.shape[0]):
            image = cv2.resize(np.squeeze(tensor[i]), (cfg['image_size'], cfg['image_size']))
            new_tensor[i] = image[:, :, np.newaxis]
        return new_tensor

    # define U-Net
    network = UNet(input_channel_count=1, output_channel_count=1, first_layer_filter_count=32)
    model = network.get_model()
    model.load_weights('unet_weights_stage_1.hdf5')

    # resize to 256 x 256

    full_images_low_res = resize_tensor(full_images)

    # prediction
    y_preds = model.predict(full_images_low_res, batch_size=16)

    kub_map = np.zeros_like(full_images)

    for i, y_pred in enumerate(y_preds):

        hm = heatmap_generate(full_images_low_res[i], y_pred)
        y_pred = cv2.resize(y_pred, (full_images[i].shape[1], full_images[i].shape[0]))
        y_pred = denormalize(y_pred, img_type='gt')
        y_pred = np.array(y_pred, dtype=np.uint8)

        # post-processing (binary + remove small cc)
        ret, y_pred_th = cv2.threshold(y_pred, 127, 255, cv2.THRESH_BINARY)
        y_pred_th = remove_small_blobs(y_pred_th)

        kub_map[i] = y_pred_th[:, :, np.newaxis]

    return kub_map


def stones_segmentation(full_images, cascade=True, method=0):
    import cv2

    if cascade:

        # 1st stage: KUB generation
        if method == 0:
            kub_maps = kub_map_generation(full_images)
        else:
            kub_maps = load_images_from_folder('kub_map',
                                               image_size=cfg['full_image_size'],
                                               image_type='bw')

        images = np.zeros((len(full_images)*3, cfg['image_size'], cfg['image_size'], 1), np.float32)
        # f, subfig = plt.subplots(len(full_images), 8)
        for i, full_image in enumerate(full_images):

            ret, kub_map = cv2.threshold(kub_maps[i], 0, 255, cv2.THRESH_BINARY)

            # create 3 partitions
            l_partition, r_partition, b_partition = create_KUB_partitions(np.squeeze(full_image),
                                                                          np.squeeze(kub_map))

            l_partition = cv2.resize(l_partition, (cfg['image_size'], cfg['image_size']))
            r_partition = cv2.resize(r_partition, (cfg['image_size'], cfg['image_size']))
            b_partition = cv2.resize(b_partition, (cfg['image_size'], cfg['image_size']))

            # subfig[i, 0].imshow(denormalize(np.squeeze(full_image)), cmap='gray', vmin=0, vmax=255)
            # subfig[i, 1].imshow(denormalize(kub_map), cmap='gray', vmin=0, vmax=255)
            # subfig[i, 2].imshow(denormalize(l_partition), cmap='gray', vmin=0, vmax=255)
            # subfig[i, 4].imshow(denormalize(r_partition), cmap='gray', vmin=0, vmax=255)
            # subfig[i, 6].imshow(denormalize(b_partition), cmap='gray', vmin=0, vmax=255)

            images[3 * i] = l_partition[:, :, np.newaxis]
            images[3 * i + 1] = r_partition[:, :, np.newaxis]
            images[3 * i + 2] = b_partition[:, :, np.newaxis]

        # define U-Net parameters
        network = UNet(input_channel_count=1, output_channel_count=1, first_layer_filter_count=32)
        model = network.get_model()
        model.load_weights('unet_weights_stage_2.hdf5')

        # predict
        y_pred = model.predict(images, batch_size=16)

        # combine partitioned y_pred into full y_pred
        y_pred = np.squeeze(y_pred)
        full_y_pred = np.zeros((int(len(y_pred) / 3), cfg['full_image_size'], cfg['full_image_size']), np.float32)
        n = 0
        for i in range(0, len(y_pred), 3):
            partitions = y_pred[i:i + 3]

            # subfig[n, 3].imshow(partitions[0]*255, cmap='gray', vmin=0, vmax=255)
            # subfig[n, 5].imshow(partitions[1]*255, cmap='gray', vmin=0, vmax=255)
            # subfig[n, 7].imshow(partitions[2]*255, cmap='gray', vmin=0, vmax=255)

            full_kub_map = cv2.resize(kub_maps[n], (full_image.shape[1], full_image.shape[0]))
            ret, full_kub_map = cv2.threshold(full_kub_map, 0, 255, cv2.THRESH_BINARY)
            combined = combine_KUB_partitions(partitions[0], partitions[1], partitions[2], full_kub_map)

            # post-processing (binary + remove small cc)
            combined = denormalize(combined, img_type='gt')
            combined = np.array(combined, dtype=np.uint8)
            ret, combined = cv2.threshold(combined, 127, 255, cv2.THRESH_BINARY)
            full_y_pred[n] = remove_small_blobs(combined, min_size=10)

            n += 1

        return full_y_pred, kub_maps

    else:

        # define U-Net parameters
        network = UNet(input_channel_count=1, output_channel_count=1, first_layer_filter_count=32)
        model = network.get_model()
        model.load_weights('unet_weights_baseline.hdf5')

        # predict
        y_pred = model.predict(full_images)
        full_y_pred = np.squeeze(y_pred)

        return full_y_pred


if __name__ == '__main__':
    print(tf.__version__)

    # baseline method
    test_image = load_images_from_folder('test_images', image_size=cfg['image_size'], resize_method='fill')
    result_baseline = stones_segmentation(test_image, cascade=False)

    # proposed method
    test_image = load_images_from_folder('test_images', image_size=cfg['full_image_size'], resize_method='fill')
    results, kub_maps = stones_segmentation(test_image, cascade=True)
    plt.show()

    test_gt = load_images_from_folder('test_gt', image_size=cfg['full_image_size'], resize_method='fill')

    # f, subfig = plt.subplots(results.shape[0], 5)
    for i, result in enumerate(results):
        f, subfig = plt.subplots(1, 5)

        subfig[0].imshow(denormalize(test_image[i]), cmap='gray', vmin=0, vmax=255)
        subfig[0].axis('off')
        subfig[0].title.set_text('input')

        subfig[1].imshow(kub_maps[i], cmap='gray', vmin=0, vmax=255)
        subfig[1].axis('off')
        subfig[1].title.set_text('stone location map')

        subfig[2].imshow(denormalize(result_baseline[i], img_type='bw'), cmap='gray', vmin=0, vmax=255)
        subfig[2].axis('off')
        subfig[2].title.set_text('baseline')

        subfig[3].imshow(denormalize(result, img_type='bw'), cmap='gray', vmin=0, vmax=255)
        subfig[3].axis('off')
        subfig[3].title.set_text('cascade unet')

        subfig[4].imshow(denormalize(test_gt[i], img_type='bw'), cmap='gray', vmin=0, vmax=255)
        subfig[4].axis('off')
        subfig[4].title.set_text('gt')
        plt.show()





