import cv2
import numpy as np
import tensorflow as tf
import csv
import matplotlib.pyplot as plt

IMG_SIZE = 128

# ======================
# LOAD MODEL
# ======================
model = tf.keras.models.load_model("model_unet.h5", compile=False)

# ======================
# AMBIL PIXEL SPACING
# ======================
def get_spacing(filename):
    with open("processed/metadata.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row[0] == filename:
                return float(row[1])
    return 1.0

# ======================
# PREDICT
# ======================
def predict(image_path):
    filename = image_path.split("/")[-1]

    # load image
    img = cv2.imread(image_path, 0)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    img_input = np.expand_dims(img_resized, axis=(0, -1))

    # predict
    pred = model.predict(img_input)[0]
    mask = (pred > 0.5).astype(np.uint8)

    # ======================
    # HITUNG UKURAN
    # ======================
    area_pixel = np.sum(mask)

    spacing = get_spacing(filename)
    area_mm = area_pixel * (spacing ** 2)

    print(f"Ukuran tumor: {area_mm:.2f} mm²")

    # ======================
    # OVERLAY
    # ======================
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))

    img_color[mask_resized == 1] = [0, 0, 255]  # merah

    # tampilkan
    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.title("Original")
    plt.imshow(img, cmap='gray')

    plt.subplot(1,2,2)
    plt.title("Segmentation")
    plt.imshow(img_color)

    plt.show()

    return mask

# ======================
# TEST
# ======================
predict("processed/images/BraTS2021_00002_73.png")  