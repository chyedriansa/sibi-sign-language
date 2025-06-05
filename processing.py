import cv2
import os
import numpy as np
from imgaug import augmenters as iaa

# Fungsi untuk menghilangkan bayangan dengan metode sederhana HSV + threshold + morph
def remove_shadow(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # Threshold pada channel V (value) untuk pisahkan bayangan (gelap)
    thresh_val = 50  # bisa kamu sesuaikan
    _, mask = cv2.threshold(v, thresh_val, 255, cv2.THRESH_BINARY)
    # Morfologi untuk bersihkan noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Terapkan mask ke gambar asli, hanya bagian terang yang dipertahankan
    cleaned = cv2.bitwise_and(img, img, mask=mask)
    return cleaned

# Augmentasi dengan imgaug
seq = iaa.Sequential([
    iaa.Affine(
        rotate=(-15, 15)
    ),
    iaa.Multiply((0.8, 1.2)),  # brightness ±20%
    iaa.JpegCompression(compression=(70, 95))  # kualitas JPEG rendah antara 70-95%
])

def process_and_augment(input_folder, output_folder, augmentations_per_image=5, resize_dim=(224,224)):
    os.makedirs(output_folder, exist_ok=True)
    classes = [cls for cls in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, cls))]
    
    for cls in classes:
        cls_in = os.path.join(input_folder, cls)
        cls_out = os.path.join(output_folder, cls)
        os.makedirs(cls_out, exist_ok=True)
        
        images = [f for f in os.listdir(cls_in) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Processing class '{cls}', {len(images)} images")
        for img_name in images:
            img_path = os.path.join(cls_in, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: gagal baca {img_path}")
                continue

            # 1. Preprocessing hilangkan bayangan
            cleaned = remove_shadow(img)

            # 2. Simpan hasil preprocessing asli dulu (optional)
            base_name = os.path.splitext(img_name)[0]
            clean_path = os.path.join(cls_out, f"{base_name}_clean.jpg")
            cv2.imwrite(clean_path, cleaned)

            # 3. Augmentasi dan simpan
            for i in range(augmentations_per_image):
                augmented = seq(image=cleaned)
                # augmented = cv2.resize(augmented, resize_dim)
                aug_name = f"{base_name}_aug{i+1}.jpg"
                aug_path = os.path.join(cls_out, aug_name)
                cv2.imwrite(aug_path, augmented)

    print("Pipeline selesai.")

# --- Main ---
input_folder = "dataset/sibi-base"
output_folder = "dataset/sibi-base-aug"
process_and_augment(input_folder, output_folder)
