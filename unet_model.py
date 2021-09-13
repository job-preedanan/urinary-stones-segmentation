import tensorflow as tf


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

        # ------------------------------------------- ENCODER PART -----------------------------------------------------
        # (256 x 256 x input_channel_count)
        begin = tf.keras.Input(shape=(self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, input_channel_count))
        enc0 = self._add_convolution_block(first_layer_filter_count, begin, 'ReLU')
        enc0 = self._add_convolution_block(first_layer_filter_count, enc0, 'ReLU')
        
        # (128 x 128 x 2N)
        enc1 = self._add_encoding_layer(first_layer_filter_count * 2, enc0)

        # (64 x 64 x 4N)
        enc2 = self._add_encoding_layer(first_layer_filter_count * 4, enc1)

        # (32 x 32 x 8N)
        enc3 = self._add_encoding_layer(first_layer_filter_count * 8, enc2)

        # (16 x 16 x 16N)
        enc4 = self._add_encoding_layer(first_layer_filter_count * 16, enc3)
        enc4 = tf.keras.layers.Dropout(0.5)(enc4)

        # -------------------------------------------DECODER PART ------------------------------------------------------
        # (32 x 32 x 8N)
        dec3 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                               self.DECONV_FILTER_SIZE,
                                               strides=self.DECONV_STRIDE,
                                               kernel_initializer='he_uniform')(enc4)
        dec3 = tf.keras.layers.BatchNormalization()(dec3)
        dec3 = tf.keras.layers.Dropout(0.5)(dec3)
        dec3 = tf.keras.layers.Concatenate(axis=self.CONCATENATE_AXIS)([dec3, enc3])

        # (64 x 64 x 4N)
        dec2 = self._add_decoding_layer(first_layer_filter_count * 4, True, dec3)
        dec2 = tf.keras.layers.Concatenate(axis=self.CONCATENATE_AXIS)([dec2, enc2])

        # (128 x 128 x N)
        dec1 = self._add_decoding_layer(first_layer_filter_count * 2, True, dec2)
        dec1 = tf.keras.layers.Concatenate(axis=self.CONCATENATE_AXIS)([dec1, enc1])

        # (256 x 256 x N)
        dec0 = self._add_decoding_layer(first_layer_filter_count, True, dec1)
        dec0 = tf.keras.layers.Concatenate(axis=self.CONCATENATE_AXIS)([dec0, enc0])

        # Last layer : CONV BLOCK + convolution + Sigmoid
        last = self._add_convolution_block(first_layer_filter_count, dec0, 'ReLU')
        last = self._add_convolution_block(first_layer_filter_count, last, 'ReLU')
        last = tf.keras.layers.Conv2D(output_channel_count,
                                      kernel_size=self.CONV_FILTER_SIZE,
                                      strides=self.CONV_STRIDE,
                                      padding='same')(last)
        last = tf.keras.layers.Activation(activation='sigmoid')(last)

        self.UNET = tf.keras.Model(inputs=begin, outputs=last)

    def _add_encoding_layer(self, filter_count, sequence):

        new_sequence = tf.keras.layers.MaxPooling2D(pool_size=self.MAX_POOLING_SIZE,
                                                    strides=self.MAX_POOLING_STRIDE,
                                                    padding='same')(sequence)

        # CONV BLOCK : convolution + activation function + batch norm
        new_sequence = self._add_convolution_block(filter_count, new_sequence, 'ReLU')
        new_sequence = self._add_convolution_block(filter_count, new_sequence, 'ReLU')

        return new_sequence

    def _add_decoding_layer(self, filter_count, add_drop_layer, sequence):

        new_sequence = self._add_convolution_block(filter_count*2, sequence, 'ReLU')
        # new_sequence = self._add_convolution_block(filter_count*2, new_sequence, 'ReLU')

        # up-convolution
        new_sequence = tf.keras.layers.Conv2DTranspose(filter_count,
                                                       kernel_size=self.DECONV_FILTER_SIZE,
                                                       strides=self.DECONV_STRIDE,
                                                       kernel_initializer='he_uniform')(new_sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)

        if add_drop_layer:
            new_sequence = tf.keras.layers.Dropout(0.5)(new_sequence)
        return new_sequence

    #  CONV BLOCK : convolution + activation function + batch norm
    def _add_convolution_block(self, filter_count, sequence, activate_function):

        new_sequence = tf.keras.layers.Conv2D(filter_count,
                                              kernel_size=self.CONV_FILTER_SIZE,
                                              strides=self.CONV_STRIDE,
                                              padding='same')(sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)
        if activate_function == 'LeakyReLU':
            new_sequence = tf.keras.layers.LeakyReLU(0.2)(new_sequence)
        elif activate_function == 'ReLU':
            # new_sequence = tf.keras.layers.Activation(activation='relu')(new_sequence)
            new_sequence = tf.keras.layers.ReLU()(new_sequence)

        return new_sequence

    def get_model(self):
        return self.UNET