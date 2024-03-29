import os
import numpy as np
from matplotlib import pyplot as plt
import cv2

# Yol bilgilerini belirtin
image_folder = "./test/images/"
label_folder = "./test/labels/"
output_folder = "./test/masks/"

# Her resmi işleyin
for image_file in os.listdir(image_folder):
    if image_file.endswith(".jpg") or image_file.endswith(".png"):
        image_name = os.path.splitext(image_file)[0]
        label_file = os.path.join(label_folder, image_name + ".txt")

        # Etiket dosyasını okuyun
        with open(label_file, 'r') as f:
            lines = f.readlines()

        # Maskeleri oluşturun ve kaydedin
        for line in lines:
            parts = line.strip().split(' ')

            # Sınıf indeksini alma
            class_index = int(parts[0])

            # Koordinatları çiftler halinde düzenleme ve çizgi segmentleri oluşturma
            coordinates = list(map(float, parts[1:]))
            num_points = len(coordinates) // 2
            points = [(coordinates[i], coordinates[i + 1]) for i in range(0, len(coordinates), 2)]

            # Her bir noktayı bir sonraki ile birleştirerek mask oluşturma
            mask = np.zeros((640, 640), dtype=np.uint8)
            for i in range(num_points - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                mask = cv2.line(mask, (int(x1 * 640), int(y1 * 640)), (int(x2 * 640), int(y2 * 640)), 255, 1)

            # Son noktayı ilk nokta ile birleştirerek mask oluşturma
            x1, y1 = points[num_points - 1]
            x2, y2 = points[0]
            mask = cv2.line(mask, (int(x1 * 640), int(y1 * 640)), (int(x2 * 640), int(y2 * 640)), 255, 1)

            # Maskeleri kaydet
            output_file = os.path.join(output_folder, f"{image_name}_{class_index}.png")
            plt.imsave(output_file, mask, cmap='gray')

            print(f"Mask saved for {image_name}, class {class_index}")

print("All masks created and saved.")
