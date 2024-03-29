from ultralytics import YOLO
import cv2

model_path = './runs/segment/train/weights/last.pt'
image_path = '37_png.rf.3e419095269fe0936515e6d0fb478a9f.jpg'

img = cv2.imread('./test/images/'+image_path)
H, W, _ = img.shape

model = YOLO(model_path)

results = model(img)

for result in results:
    for j, mask in enumerate(result.masks.data):
        mask = mask.numpy() * 255
        mask = cv2.resize(mask, (W, H))
        cv2.imwrite(f'./test/images/{image_path}.png', mask)