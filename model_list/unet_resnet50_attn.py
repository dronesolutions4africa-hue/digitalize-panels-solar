"""
U-Net ResNet50 with Attention Gates + Deep Supervision.

Attention gates (Oktay et al. 2018) let the decoder focus on panel regions
by computing a per-pixel attention map from skip + gating signals.

Deep supervision adds auxiliary segmentation heads at /4 and /8 scales so
gradients flow to early decoder layers even before the final output.
"""
import tensorflow as tf


def _double_conv(x, filters, dropout=0.0):
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    if dropout > 0:
        x = tf.keras.layers.SpatialDropout2D(dropout)(x)
    return x


def _attention_gate(skip, gating, inter_channels):
    """
    Attention gate: modulate skip connection using gating signal.
    skip    — feature map from encoder at resolution R
    gating  — feature map from coarser decoder stage (R/2)
    Returns attention-weighted skip connection.
    """
    theta = tf.keras.layers.Conv2D(inter_channels, 1, padding='same', use_bias=False)(skip)
    phi   = tf.keras.layers.Conv2D(inter_channels, 1, padding='same', use_bias=False)(gating)
    phi   = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')(phi)
    add   = tf.keras.layers.Add()([theta, phi])
    add   = tf.keras.layers.ReLU()(add)
    psi   = tf.keras.layers.Conv2D(1, 1, padding='same', use_bias=False)(add)
    psi   = tf.keras.layers.Activation('sigmoid')(psi)
    return tf.keras.layers.Multiply()([skip, psi])


def _upsample_attn_concat(x, skip, filters, inter_channels, dropout=0.0):
    attn_skip = _attention_gate(skip, x, inter_channels)
    x = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')(x)
    x = tf.keras.layers.Concatenate()([x, attn_skip])
    return _double_conv(x, filters, dropout=dropout)


def build_unet_resnet50_attn(input_shape=(512, 512, 3), n_labels=2,
                              encoder_weights='imagenet', dropout=0.1):
    """
    Attention U-Net with ResNet50 encoder and deep supervision.

    Returns (model, encoder, aux_outputs) where aux_outputs is a list of
    auxiliary segmentation heads (used for deep supervision loss).

    Deep supervision outputs (auxiliary heads — same n_labels, softmax):
        aux_d3  — at /4  resolution (H/4 × W/4)
        aux_d4  — at /8  resolution (H/8 × W/8)

    Training loss = main_loss + 0.4 * aux_d3_loss + 0.2 * aux_d4_loss
    """
    inputs = tf.keras.Input(shape=input_shape, name='input')

    encoder = tf.keras.applications.ResNet50(
        include_top=False,
        weights=encoder_weights,
        input_tensor=inputs,
    )

    s1 = encoder.get_layer('conv1_relu').output           # /2   H/2  × W/2,   64ch
    s2 = encoder.get_layer('conv2_block3_out').output     # /4   H/4  × W/4,  256ch
    s3 = encoder.get_layer('conv3_block4_out').output     # /8   H/8  × W/8,  512ch
    s4 = encoder.get_layer('conv4_block6_out').output     # /16  H/16 × W/16, 1024ch
    bridge = encoder.get_layer('conv5_block3_out').output # /32  H/32 × W/32, 2048ch

    # Decoder with attention gates
    d1 = _upsample_attn_concat(bridge, s4, 512, 256, dropout=dropout)   # /32 → /16
    d2 = _upsample_attn_concat(d1,     s3, 256, 128, dropout=dropout)   # /16 → /8
    d3 = _upsample_attn_concat(d2,     s2, 128,  64, dropout=dropout)   # /8  → /4
    d4 = _upsample_attn_concat(d3,     s1,  64,  32, dropout=dropout)   # /4  → /2

    x = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')(d4)
    x = _double_conv(x, 32, dropout=0.0)

    # Main output (full resolution) — dtype=float32 ensures stable loss with mixed precision
    main_out = tf.keras.layers.Conv2D(
        n_labels, 1, padding='same', activation='softmax',
        dtype='float32', name='main_output'
    )(x)

    # Auxiliary output at /4 (d3) — upsampled to full res for loss computation
    aux_d3_map = tf.keras.layers.Conv2D(n_labels, 1, padding='same', activation='softmax',
                                         dtype='float32', name='aux_d3_logits')(d3)
    aux_d3 = tf.keras.layers.UpSampling2D(
        (input_shape[0] // aux_d3_map.shape[1], input_shape[1] // aux_d3_map.shape[2]),
        interpolation='bilinear', name='aux_d3_output'
    )(aux_d3_map)

    # Auxiliary output at /8 (d2) — upsampled to full res
    aux_d4_map = tf.keras.layers.Conv2D(n_labels, 1, padding='same', activation='softmax',
                                         dtype='float32', name='aux_d4_logits')(d2)
    aux_d4 = tf.keras.layers.UpSampling2D(
        (input_shape[0] // aux_d4_map.shape[1], input_shape[1] // aux_d4_map.shape[2]),
        interpolation='bilinear', name='aux_d4_output'
    )(aux_d4_map)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=[main_out, aux_d3, aux_d4],
        name='AttentionUNet_ResNet50'
    )
    return model, encoder
