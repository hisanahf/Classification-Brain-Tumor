import os
import nibabel as nib
import numpy as np
import cv2
import csv
from tqdm import tqdm

# PATH DATASET (PASTIKAN SESUAI)
INPUT_DIR = "brats_sample"
IMG_OUT = "processed/images"
MASK_OUT = "processed/masks"
METADATA = "processed/metadata.csv"

os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(MASK_OUT, exist_ok=True)

IMG_SIZE = 128

with open(METADATA, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "pixel_spacing"])

    for patient in tqdm(os.listdir(INPUT_DIR)):
        patient_path = os.path.join(INPUT_DIR, patient)

        try:
            flair_path = os.path.join(patient_path, f"{patient}_flair.nii.gz")
            seg_path = os.path.join(patient_path, f"{patient}_seg.nii.gz")

            flair = nib.load(flair_path).get_fdata()
            seg = nib.load(seg_path).get_fdata()

            # ambil pixel spacing (mm)
            spacing = nib.load(flair_path).header.get_zooms()[0]

            for i in range(flair.shape[2]):
                img = flair[:, :, i]
                mask = seg[:, :, i]

                if np.max(mask) == 0:
                    continue  # skip kalau tidak ada tumor

                # normalisasi
                img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

                # resize
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

                filename = f"{patient}_{i}.png"

                cv2.imwrite(os.path.join(IMG_OUT, filename), img * 255)
                cv2.imwrite(os.path.join(MASK_OUT, filename), mask)

                writer.writerow([filename, spacing])

        except Exception as e:
            print(f"Error di {patient}: {e}")