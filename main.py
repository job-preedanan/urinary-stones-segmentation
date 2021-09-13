import tensorflow as tf
import numpy as np
import cv2
import os
from unet_model import UNet
from utils import create_KUB_partitions, combine_KUB_partitions, normalize, denormalize, remove_small_blobs
import matplotlib.pyplot as plt


# define configuration
cfg = {'image_size': 256,
       'full_image_size': 1024,
       'cascade_unet': True}


def load_images_from_folder(folder_name):
    images_list = os.listdir(folder_name)
    images = np.zeros((len(images_list), cfg['image_size'], cfg['image_size'], 1))
    for i, image_name in enumerate(images_list):
        image = cv2.imread(folder_name + os.sep + image_name, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (cfg['image_size'], cfg['image_size']))
        image = image[:, :, np.newaxis]
        images[i] = normalize(image)

    return images


# 1st-stage : KUB map generation
def kub_map_generation(full_images):

    # U-Net define
    network = UNet(input_channel_count=1, output_channel_count=1, first_layer_filter_count=32)
    model = network.get_model()
    model.load_weights('unet_weights_stage_1.hdf5')

    # prediction
    y_preds = model.predict(full_images, batch_size=16)

    kub_map = np.zeros_like(y_preds)
    for i, y_pred in enumerate(y_preds):
        y_pred = cv2.resize(y_pred, (full_images[i].shape[1], full_images[i].shape[0]))
        y_pred = denormalize(y_pred, img_type='gt')
        y_pred = np.array(y_pred, dtype=np.uint8)

        # post-processing (binary + remove small contours)
        ret, y_pred_th = cv2.threshold(y_pred, 127, 255, cv2.THRESH_BINARY)
        y_pred_th = remove_small_blobs(y_pred_th)

        kub_map[i] = y_pred_th[:, :, np.newaxis]

    return kub_map


def stones_segmentation(full_images, cascade=True):
    import cv2

    if cascade:

        # 1st stage: KUB generation
        kub_maps = kub_map_generation(full_images)
        images = np.zeros((len(full_images)*3, cfg['image_size'], cfg['image_size'], 1))
        f, subfig = plt.subplots(len(full_images), 7)
        for i, full_image in enumerate(full_images):

            # create 3 partitions
            l_partition, r_partition, b_partition = create_KUB_partitions(np.squeeze(full_image),
                                                                          np.squeeze(kub_maps[i]))

            l_partition = cv2.resize(l_partition, (cfg['image_size'], cfg['image_size']))
            r_partition = cv2.resize(r_partition, (cfg['image_size'], cfg['image_size']))
            b_partition = cv2.resize(b_partition, (cfg['image_size'], cfg['image_size']))

            subfig[i, 0].imshow(denormalize(np.squeeze(full_image)), cmap='gray', vmin=0, vmax=255)
            subfig[i, 1].imshow(denormalize(l_partition), cmap='gray', vmin=0, vmax=255)
            subfig[i, 3].imshow(denormalize(r_partition), cmap='gray', vmin=0, vmax=255)
            subfig[i, 5].imshow(denormalize(b_partition), cmap='gray', vmin=0, vmax=255)

            images[3 * i] = l_partition[:, :, np.newaxis]
            images[3 * i + 1] = r_partition[:, :, np.newaxis]
            images[3 * i + 2] = b_partition[:, :, np.newaxis]

        # define U-Net parameters
        network = UNet(input_channel_count=1, output_channel_count=1, first_layer_filter_count=32)
        model = network.get_model()
        model.load_weights('unet_weights_stage_2.hdf5')

        # predict
        y_pred = model.predict(images)

        # combine partitioned y_pred into full y_pred
        y_pred = np.squeeze(y_pred)
        full_y_pred = np.zeros((int(len(y_pred) / 3), cfg['full_image_size'], cfg['full_image_size']), np.float32)
        n = 0
        for i in range(0, len(y_pred), 3):
            partitions = y_pred[i:i + 3]

            subfig[n, 2].imshow(denormalize(partitions[0]), cmap='gray', vmin=0, vmax=255)
            subfig[n, 4].imshow(denormalize(partitions[1]), cmap='gray', vmin=0, vmax=255)
            subfig[n, 6].imshow(denormalize(partitions[2]), cmap='gray', vmin=0, vmax=255)

            full_kub_map = cv2.resize(kub_maps[n], (cfg['full_image_size'], cfg['full_image_size']))
            full_y_pred[n] = combine_KUB_partitions(partitions[0], partitions[1], partitions[2], full_kub_map)
            n += 1

        plt.show()

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

    test_image = load_images_from_folder('test_images')
    test_gt = load_images_from_folder('test_gt')
    results, kub_maps = stones_segmentation(test_image, cascade=True)
    result_baseline = stones_segmentation(test_image, cascade=False)
    print(results.shape)

    f, subfig = plt.subplots(results.shape[0], 5)
    for i, result in enumerate(results):
        subfig[i, 0].imshow(denormalize(test_image[i]), cmap='gray', vmin=0, vmax=255)
        subfig[i, 0].axis('off')
        subfig[i, 0].title.set_text('input')

        subfig[i, 1].imshow(kub_maps[i], cmap='gray', vmin=0, vmax=255)
        subfig[i, 1].axis('off')
        subfig[i, 1].title.set_text('stone location map')

        subfig[i, 2].imshow(denormalize(result_baseline[i], img_type='bw'), cmap='gray', vmin=0, vmax=255)
        subfig[i, 2].axis('off')
        subfig[i, 2].title.set_text('baseline')

        subfig[i, 3].imshow(denormalize(result, img_type='bw'), cmap='gray', vmin=0, vmax=255)
        subfig[i, 3].axis('off')
        subfig[i, 3].title.set_text('casecade unet')

        subfig[i, 4].imshow(denormalize(test_gt[i], img_type='bw'), cmap='gray', vmin=0, vmax=255)
        subfig[i, 4].axis('off')
        subfig[i, 4].title.set_text('gt')
    plt.show()





