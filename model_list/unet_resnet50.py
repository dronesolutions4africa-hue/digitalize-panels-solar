import tensorflow as tf


def _double_conv(x, filters):
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def _upsample_concat(x, skip, filters):
    x = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')(x)
    x = tf.keras.layers.Concatenate()([x, skip])
    return _double_conv(x, filters)


def build_unet_resnet50(input_shape=(512, 512, 3), n_labels=2):
    """
    U-Net with ResNet50 encoder pretrained on ImageNet.
    Returns (model, encoder) so the caller can freeze/unfreeze the encoder.

    Skip connections at each decoder stage:
        conv1_relu        /2  → H/2  × W/2,   64 ch
        conv2_block3_out  /4  → H/4  × W/4,  256 ch
        conv3_block4_out  /8  → H/8  × W/8,  512 ch
        conv4_block6_out  /16 → H/16 × W/16, 1024 ch
        conv5_block3_out  /32 → H/32 × W/32, 2048 ch  (bottleneck)
    """
    inputs = tf.keras.Input(shape=input_shape, name='input')

    encoder = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs,
    )

    s1 = encoder.get_layer('conv1_relu').output         # /2
    s2 = encoder.get_layer('conv2_block3_out').output   # /4
    s3 = encoder.get_layer('conv3_block4_out').output   # /8
    s4 = encoder.get_layer('conv4_block6_out').output   # /16
    bridge = encoder.get_layer('conv5_block3_out').output  # /32

    x = _upsample_concat(bridge, s4, 512)   # /32 → /16
    x = _upsample_concat(x,      s3, 256)   # /16 → /8
    x = _upsample_concat(x,      s2, 128)   # /8  → /4
    x = _upsample_concat(x,      s1, 64)    # /4  → /2

    x = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')(x)
    x = _double_conv(x, 32)

    outputs = tf.keras.layers.Conv2D(
        n_labels, 1, padding='same', activation='softmax', name='output'
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='UNet_ResNet50')
    return model, encoder
