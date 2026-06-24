import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = 128

# ======================
# LOAD DATA
# ======================
def load_data():
    image_paths = sorted(glob("processed/images/*.png"))
    mask_paths = sorted(glob("processed/masks/*.png"))

    images = []
    masks = []

    for img_path, mask_path in zip(image_paths, mask_paths):
        # image
        img = cv2.imread(img_path, 0) / 255.0

        # 🔥 FIX: mask jadi binary
        mask = cv2.imread(mask_path, 0)
        mask = (mask > 0).astype(np.float32)

        img = np.expand_dims(img, axis=-1)
        mask = np.expand_dims(mask, axis=-1)

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)

# ======================
# DICE COEFFICIENT
# ======================
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )

# ======================
# DICE LOSS
# ======================
def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

# ======================
# 🔥 COMBINED LOSS (LEBIH KUAT)
# ======================
def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

# ======================
# IOU (JACCARD)
# ======================
def iou_metric(y_true, y_pred):
    smooth = 1.

    # 🔥 threshold biar realistis
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection

    return (intersection + smooth) / (union + smooth)

# ======================
# MODEL U-NET
# ======================
def unet():
    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 1))

    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(c3)

    u1 = layers.UpSampling2D()(c3)
    u1 = layers.concatenate([u1, c2])
    c4 = layers.Conv2D(32, 3, activation='relu', padding='same')(u1)

    u2 = layers.UpSampling2D()(c4)
    u2 = layers.concatenate([u2, c1])
    c5 = layers.Conv2D(16, 3, activation='relu', padding='same')(u2)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c5)

    return models.Model(inputs, outputs)

# ======================
# TRAIN
# ======================
X, y = load_data()

print("Data shape:", X.shape)

model = unet()

model.compile(
    optimizer='adam',
    loss=combined_loss,  # 🔥 pakai loss baru
    metrics=[dice_coef, iou_metric, 'accuracy']
)

model.fit(
    X, y,
    epochs=30,          # 🔥 tambah epoch
    batch_size=8,
    validation_split=0.1
)

model.save("model_unet.h5")