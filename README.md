## CASCADED UNET PIPELINE FOR URINARY STONE SEGMENTATION

in this WORK, we proposed a two-stage pipeline for segmenting urinary stones. The frst stage U-Net generated the map localizing the urinary organs in full abdominal x-ray images
Then, this map was used for creating partitioned images input to the second stage U-Net to reduce class imbalance and was also used in stone-embedding augmentation to increase a number of training data. The U-Net model was trained with the combination of real stone-contained images and synthesized stone-embedded images to segment urinary stones on the partitioned input images.

<a href="https://ibb.co/w7FT2tF"><img src="https://i.ibb.co/XDr9GHr/pipeline-urinary-stones-work.png" alt="pipeline-urinary-stones-work" border="0"></a>

Fig. 1 The overview of proposed pipeline for segmenting urinary stones. The 1st stage U-Net generates KUB region maps from downsampled abdominal
x-ray images. The results from this stage are upsampled and used for stone-embedding augmentation, and cropping a full image into 3 partitions based on the
anatomical region. The 2nd stage U-Net processes the partitioned images and generates the segmented stones results. Post-processing consists of the detection
of false bladder stones and the removal of lesions outside the stone localization map.

# DATASET

<a href="https://ibb.co/xshn8Lc"><img src="https://i.ibb.co/Db4BgW6/dataset.png" alt="dataset" border="0"></a>

Fig. 2 Illustration of an abdominal x-ray image with stones (left), corresponding gold standard manual segmentation of the stones (middle) and a stone-free abdominal x-ray image (right
