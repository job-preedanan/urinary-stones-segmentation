import tensorflow as tf


# This code is for creating the Unet-based model such as Unet, ResUnet, and attention Unet
class UNet(object):
    def __init__(self, input_channel_count, output_channel_count, first_layer_filter_count):
        self.INPUT_IMAGE_SIZE = 256

        # encoder parameters
        self.CONV_FILTER_SIZE = 3
        self.CONV_STRIDE = 1
        self.CONV_PADDING = (1, 1)
        self.MAX_POOLING_SIZE = 3
        self.MAX_POOLING_STRIDE = 2
        self.MAX_POOLING_PADDING = (1, 1)

        # decoder parameters
        self.DECONV_FILTER_SIZE = 2
        self.DECONV_STRIDE = 2
        self.CONCATENATE_AXIS = -1

        # filters
        self.nb_filters = [first_layer_filter_count,
                           first_layer_filter_count * 2,
                           first_layer_filter_count * 4,
                           first_layer_filter_count * 8,
                           first_layer_filter_count * 16]

        self.attention = False

        # ------------------------------------------- ENCODER PART -----------------------------------------------------
        # (256 x 256 x input_channel_count)
        begin = tf.keras.Input(shape=(self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, input_channel_count))
        enc0 = self._add_convolution_block(self.nb_filters[0], 2, begin, residual=False)

        # (128 x 128 x 2N)
        enc1 = tf.keras.layers.MaxPooling2D(pool_size=self.MAX_POOLING_SIZE,
                                            strides=self.MAX_POOLING_STRIDE,
                                            padding='same')(enc0)
        enc1 = self._add_convolution_block(self.nb_filters[1], 2, enc1, residual=False)

        # (64 x 64 x 4N)
        enc2 = tf.keras.layers.MaxPooling2D(pool_size=self.MAX_POOLING_SIZE,
                                            strides=self.MAX_POOLING_STRIDE,
                                            padding='same')(enc1)
        enc2 = self._add_convolution_block(self.nb_filters[2], 2, enc2, residual=False)

        # (32 x 32 x 8N)
        enc3 = tf.keras.layers.MaxPooling2D(pool_size=self.MAX_POOLING_SIZE,
                                            strides=self.MAX_POOLING_STRIDE,
                                            padding='same')(enc2)
        enc3 = self._add_convolution_block(self.nb_filters[3], 2, enc3, residual=False)

        # (16 x 16 x 16N)
        enc4 = tf.keras.layers.MaxPooling2D(pool_size=self.MAX_POOLING_SIZE,
                                            strides=self.MAX_POOLING_STRIDE,
                                            padding='same')(enc3)
        enc4 = self._add_convolution_block(self.nb_filters[4], 2,  enc4, residual=False)
        enc4 = tf.keras.layers.Dropout(0.5)(enc4)

        # -------------------------------------------DECODER PART ------------------------------------------------------
        # (32 x 32 x 8N)
        dec3 = self._add_upsampling_layer(self.nb_filters[3], True, enc4)
        if self.attention:
            enc3 = self._add_attention_block(enc3, enc4, self.nb_filters[3])
        dec3 = tf.keras.layers.Concatenate(axis=self.CONCATENATE_AXIS)([dec3, enc3])
        dec3 = self._add_convolution_block(self.nb_filters[3], 1, dec3, residual=False)

        # (64 x 64 x 4N)
        dec2 = self._add_upsampling_layer(first_layer_filter_count * 4, True, dec3)
        if self.attention:
            enc2 = self._add_attention_block(enc2, dec3, self.nb_filters[2])
        dec2 = tf.keras.layers.Concatenate(axis=self.CONCATENATE_AXIS)([dec2, enc2])
        dec2 = self._add_convolution_block(self.nb_filters[2], 1, dec2, residual=False)

        # (128 x 128 x N)
        dec1 = self._add_upsampling_layer(first_layer_filter_count * 2, True, dec2)
        if self.attention:
            enc1 = self._add_attention_block(enc1, dec2, self.nb_filters[1])
        dec1 = tf.keras.layers.Concatenate(axis=self.CONCATENATE_AXIS)([dec1, enc1])
        dec1 = self._add_convolution_block(self.nb_filters[1], 1, dec1, residual=False)

        # (256 x 256 x N)
        dec0 = self._add_upsampling_layer(first_layer_filter_count, True, dec1)
        if self.attention:
            enc0 = self._add_attention_block(enc0, dec1, self.nb_filters[0])
        dec0 = tf.keras.layers.Concatenate(axis=self.CONCATENATE_AXIS)([dec0, enc0])
        dec0 = self._add_convolution_block(self.nb_filters[0], 2, dec0, residual=False)

        # Last layer : CONV BLOCK + convolution + Sigmoid
        last = tf.keras.layers.Conv2D(output_channel_count,
                                      kernel_size=self.CONV_FILTER_SIZE,
                                      strides=self.CONV_STRIDE,
                                      padding='same')(dec0)
        last = tf.keras.layers.Activation(activation='sigmoid')(last)

        self.UNET = tf.keras.Model(inputs=begin, outputs=last)

    def _add_convolution_block(self, filter_count, conv_count, sequence, residual=False):

        new_sequence = sequence
        for n in range(conv_count):
            new_sequence = self._add_convolution_layer(filter_count, new_sequence, 'ReLU')

        if residual:
            shortcut = tf.keras.layers.Conv2D(filter_count,
                                              kernel_size=1,
                                              strides=self.CONV_STRIDE,
                                              padding='same')(sequence)
            new_sequence = tf.keras.layers.add([shortcut, new_sequence])

        return new_sequence

    def _add_upsampling_layer(self, filter_count, add_drop_layer, sequence):

        # up-convolution
        new_sequence = tf.keras.layers.Conv2DTranspose(filter_count,
                                                       kernel_size=self.DECONV_FILTER_SIZE,
                                                       strides=self.DECONV_STRIDE,
                                                       kernel_initializer='he_uniform')(sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)

        if add_drop_layer:
            new_sequence = tf.keras.layers.Dropout(0.5)(new_sequence)
        return new_sequence

    #  CONV BLOCK : convolution + activation function + batch norm
    def _add_convolution_layer(self, filter_count, sequence, activate_function):

        new_sequence = tf.keras.layers.Conv2D(filter_count,
                                              kernel_size=self.CONV_FILTER_SIZE,
                                              strides=self.CONV_STRIDE,
                                              padding='same')(sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)
        if activate_function == 'LeakyReLU':
            new_sequence = tf.keras.layers.LeakyReLU(0.2)(new_sequence)
        elif activate_function == 'ReLU':
            new_sequence = tf.keras.layers.ReLU()(new_sequence)

        return new_sequence

    def _add_attention_block(self, sequence, gating, filter_count):

        # reshape x and gating layers
        x = tf.keras.layers.Conv2D(filter_count, kernel_size=1, strides=2, padding='same')(sequence)
        gating = tf.keras.layers.Conv2D(filter_count, kernel_size=1, strides=1, padding='same')(gating)

        # add x and g + ReLU
        concate_xg = tf.keras.layers.add([x, gating])
        relu_xg = tf.keras.layers.ReLU()(concate_xg)

        # conv(nb_filter=1) + sigmoid
        conv1_xg = tf.keras.layers.Conv2D(1, kernel_size=1, strides=1, padding='same')(relu_xg)
        sigmoid_xg = tf.keras.layers.Activation(activation='sigmoid')(conv1_xg)

        # upsample
        upsample_sigmoid_xg = tf.keras.layers.UpSampling2D(size=(2, 2))(sigmoid_xg)

        # multiply with x and conv(nb_filters=x_filters)
        y = tf.keras.layers.multiply([upsample_sigmoid_xg, sequence])
        conv1_xg = tf.keras.layers.Conv2D(filter_count, kernel_size=1, strides=1, padding='same')(y)
        new_sequence = tf.keras.layers.BatchNormalization()(conv1_xg)

        return new_sequence

    def get_model(self):
        return self.UNET


class MultiResUnet(object):
    def __init__(self, input_channel_count, output_channel_count):
        self.INPUT_IMAGE_SIZE = 256

        # encoder parameters
        self.CONV_STRIDE = 1
        self.MAX_POOLING_SIZE = 3
        self.MAX_POOLING_STRIDE = 2

        # decoder parameters
        self.DECONV_FILTER_SIZE = 2
        self.DECONV_STRIDE = 2
        self.CONCATENATE_AXIS = -1

        # encoder
        inputs = tf.keras.Input(shape=(self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, input_channel_count))

        multi_res_block1 = self._add_multi_res_block(first_filter_count=16, input_sequence=inputs, activation='relu')
        res_path1 = self._add_res_path_block(filter_count=32, conv_count=4, input_sequence=multi_res_block1)

        multi_res_block2 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(multi_res_block1)
        multi_res_block2 = self._add_multi_res_block(first_filter_count=32, input_sequence=multi_res_block2,
                                                     activation='relu')
        res_path2 = self._add_res_path_block(filter_count=64, conv_count=3, input_sequence=multi_res_block2)

        multi_res_block3 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(multi_res_block2)
        multi_res_block3 = self._add_multi_res_block(first_filter_count=64, input_sequence=multi_res_block3,
                                                     activation='relu')
        res_path3 = self._add_res_path_block(filter_count=128, conv_count=2, input_sequence=multi_res_block3)

        multi_res_block4 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(multi_res_block3)
        multi_res_block4 = self._add_multi_res_block(first_filter_count=128, input_sequence=multi_res_block4,
                                                     activation='relu')
        res_path4 = self._add_res_path_block(filter_count=256, conv_count=1, input_sequence=multi_res_block4)

        multi_res_block5 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(multi_res_block4)
        multi_res_block5 = self._add_multi_res_block(first_filter_count=256, input_sequence=multi_res_block5,
                                                     activation='relu')

        # decoder
        multi_res_block6 = self._add_upsampling_layer(filter_count=256, input_sequence=multi_res_block5)
        multi_res_block6 = tf.keras.layers.Concatenate(axis=-1)([multi_res_block6, res_path4])
        multi_res_block6 = self._add_multi_res_block(first_filter_count=128, input_sequence=multi_res_block6,
                                                     activation='relu')

        multi_res_block7 = self._add_upsampling_layer(filter_count=128, input_sequence=multi_res_block6)
        multi_res_block7 = tf.keras.layers.Concatenate(axis=-1)([multi_res_block7, res_path3])
        multi_res_block7 = self._add_multi_res_block(first_filter_count=64, input_sequence=multi_res_block7,
                                                     activation='relu')

        multi_res_block8 = self._add_upsampling_layer(filter_count=64, input_sequence=multi_res_block7)
        multi_res_block8 = tf.keras.layers.Concatenate(axis=-1)([multi_res_block8, res_path2])
        multi_res_block8 = self._add_multi_res_block(first_filter_count=32, input_sequence=multi_res_block8,
                                                     activation='relu')

        multi_res_block9 = self._add_upsampling_layer(filter_count=32, input_sequence=multi_res_block8)
        multi_res_block9 = tf.keras.layers.Concatenate(axis=-1)([multi_res_block9, res_path1])
        multi_res_block9 = self._add_multi_res_block(first_filter_count=16, input_sequence=multi_res_block9,
                                                     activation='relu')

        # output
        outputs = tf.keras.layers.Conv2D(filters=output_channel_count,
                                         kernel_size=3,
                                         strides=1,
                                         padding='same')(multi_res_block9)
        outputs = tf.keras.layers.Activation(activation='sigmoid')(outputs)

        self.MultiResUnet = tf.keras.Model(inputs=inputs, outputs=outputs)

    def _add_multi_res_block(self, first_filter_count, input_sequence, activation):

        # 3 convolution layers
        layer1 = self._add_convolution_layer(first_filter_count, input_sequence, kernel_size=3,
                                             activation=activation)
        layer2 = self._add_convolution_layer(first_filter_count*2, layer1, kernel_size=3, activation=activation)
        layer3 = self._add_convolution_layer(first_filter_count*4, layer2, kernel_size=3, activation=activation)

        # concatenate
        concatenate_layer = tf.keras.layers.Concatenate(axis=-1)([layer3, layer2])
        concatenate_layer = tf.keras.layers.Concatenate(axis=-1)([concatenate_layer, layer1])

        # add with residual layer
        total_filter_count = first_filter_count + first_filter_count * 2  + first_filter_count * 4
        shortcut = self._add_convolution_layer(total_filter_count, input_sequence, kernel_size=1, activation='relu')
        new_sequence = tf.keras.layers.add([shortcut, concatenate_layer])

        return new_sequence

    def _add_res_path_block(self, filter_count, conv_count, input_sequence):

        for n in range(conv_count):
            new_sequence = self._add_convolution_layer(filter_count, input_sequence, kernel_size=3, activation='relu')
            shortcut = self._add_convolution_layer(filter_count, input_sequence, kernel_size=1, activation='relu')
            new_sequence = tf.keras.layers.add([shortcut, new_sequence])

        return new_sequence

    def _add_convolution_layer(self, filter_count, sequence, kernel_size, activation):
        sequence = tf.keras.layers.Conv2D(filter_count,
                                          kernel_size=kernel_size,
                                          strides=self.CONV_STRIDE,
                                          padding='same')(sequence)
        sequence = tf.keras.layers.BatchNormalization()(sequence)
        if activation == 'relu':
            new_sequence = tf.keras.layers.ReLU()(sequence)
        elif activation == 'leakyrelu':
            new_sequence = tf.keras.layers.LeakyReLU(0.2)(sequence)

        return new_sequence

    def _add_upsampling_layer(self, filter_count, input_sequence):
        new_sequence = tf.keras.layers.Conv2DTranspose(filter_count,
                                                       kernel_size=self.DECONV_FILTER_SIZE,
                                                       strides=self.DECONV_STRIDE,
                                                       kernel_initializer='he_uniform')(input_sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)
        new_sequence = tf.keras.layers.Dropout(0.5)(new_sequence)  #
        return new_sequence

    def get_model(self):
        return self.MultiResUnet

    
model = UNet(1, 1, 32).get_model()
model.summary()