# import cv2
# import mediapipe as mp
# import numpy as np

# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# image_path = "dataset/sibi/A/IMG_E9405.JPG"
# image = cv2.imread(image_path)
# if image is None:
#     print("Gagal membaca gambar:", image_path)
#     exit()

# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# results = hands.process(image_rgb)

# if not results.multi_hand_landmarks:
#     print("Tidak ada tangan terdeteksi.")
#     exit()

# hand_landmarks = results.multi_hand_landmarks[0]

# h, w, _ = image.shape
# x_coords = [lm.x for lm in hand_landmarks.landmark]
# y_coords = [lm.y for lm in hand_landmarks.landmark]

# padding = 200
# x_min = int(min(x_coords) * w) - padding
# x_max = int(max(x_coords) * w) + padding
# y_min = int(min(y_coords) * h) - padding
# y_max = int(max(y_coords) * h) + padding

# x_min = max(x_min, 0)
# y_min = max(y_min, 0)
# x_max = min(x_max, w)
# y_max = min(y_max, h)

# hand_crop = image[y_min:y_max, x_min:x_max]
# crop_h, crop_w = hand_crop.shape[:2]
# white_bg = 255 * np.ones_like(hand_crop)

# hand_landmarks_shifted = []
# for lm in hand_landmarks.landmark:
#     x = int(lm.x * w) - x_min
#     y = int(lm.y * h) - y_min
#     hand_landmarks_shifted.append((x, y))

# connections = mp_hands.HAND_CONNECTIONS
# line_color = (0, 255, 0)  # Merah
# line_thickness = 5
# circle_radius = 8

# for connection in connections:
#     start_idx = connection[0]
#     end_idx = connection[1]
#     start_point = hand_landmarks_shifted[start_idx]
#     end_point = hand_landmarks_shifted[end_idx]
#     cv2.line(white_bg, start_point, end_point, line_color, line_thickness)

# for point in hand_landmarks_shifted:
#     cv2.circle(white_bg, point, circle_radius, line_color, -1)

# cv2.imshow("Hand Skeleton on White Background", white_bg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



import os
import cv2
import mediapipe as mp
import numpy as np

# Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Path input & output
input_root = "dataset/sibi-base-aug"
output_root = "dataset/skeleton-aug"

# Parameter visualisasi
padding = 200
line_color = (0, 0, 255)  # Merah (BGR)
line_thickness = 10
circle_radius = 8

# Loop tiap huruf (folder A-Z)
for label in os.listdir(input_root):
    label_path = os.path.join(input_root, label)
    if not os.path.isdir(label_path):
        continue

    # Buat folder output jika belum ada
    output_label_path = os.path.join(output_root, label)
    os.makedirs(output_label_path, exist_ok=True)

    # Loop semua gambar di folder label
    for file_name in os.listdir(label_path):
        file_path = os.path.join(label_path, file_name)

        # Baca gambar
        image = cv2.imread(file_path)
        if image is None:
            print(f"❌ Gagal membaca: {file_path}")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            print(f"🖐️ Tidak ada tangan: {file_path}")
            continue

        hand_landmarks = results.multi_hand_landmarks[0]

        h, w, _ = image.shape
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]

        x_min = int(min(x_coords) * w) - padding
        x_max = int(max(x_coords) * w) + padding
        y_min = int(min(y_coords) * h) - padding
        y_max = int(max(y_coords) * h) + padding

        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_max, w)
        y_max = min(y_max, h)

        hand_crop = image[y_min:y_max, x_min:x_max]
        crop_h, crop_w = hand_crop.shape[:2]
        white_bg = 255 * np.ones_like(hand_crop)

        hand_landmarks_shifted = []
        for lm in hand_landmarks.landmark:
            x = int(lm.x * w) - x_min
            y = int(lm.y * h) - y_min
            hand_landmarks_shifted.append((x, y))

        connections = mp_hands.HAND_CONNECTIONS

        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            start_point = hand_landmarks_shifted[start_idx]
            end_point = hand_landmarks_shifted[end_idx]
            cv2.line(white_bg, start_point, end_point, line_color, line_thickness)

        for point in hand_landmarks_shifted:
            cv2.circle(white_bg, point, circle_radius, line_color, -1)

        # Simpan hasil
        output_path = os.path.join(output_label_path, file_name)
        cv2.imwrite(output_path, white_bg)
        # print(f"✅ Disimpan: {output_path}")

print("✅ Proses selesai!")