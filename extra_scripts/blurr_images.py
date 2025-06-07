import os
import cv2
import numpy as np
import easyocr
from pathlib import Path

# Set your folders here
input_folder = Path("downloaded_images_multimodal")
output_folder = input_folder / "blurred_easyocr"
output_folder.mkdir(exist_ok=True)

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

# Loop through all images
for image_file in input_folder.glob("*.jpg"):
    image = cv2.imread(str(image_file))
    if image is None:
        continue

    results = reader.readtext(image)
    
    for bbox, text, conf in results:
        if conf < 0.4:
            print('Not sure about this text: ', text, 'confidence:', conf)
            print('Still blurring...')

        pts = cv2.boundingRect(cv2.convexHull(np.array(bbox).astype(int)))
        x, y, w, h = pts
        roi = image[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        blur = cv2.GaussianBlur(roi, (55, 55), 0)
        image[y:y+h, x:x+w] = blur

    out_path = output_folder / image_file.name
    cv2.imwrite(str(out_path), image)
    print(f"Blurred and saved: {out_path}")
