import os
import requests

# List of image URLs
urls = [
    "https://uva.fra1.qualtrics.com/CP/Graphic.php?IM=IM_3ly3QjRjF9wU6vc",
    "https://uva.fra1.qualtrics.com/CP/Graphic.php?IM=IM_b2vqHITA9jITeuy",
    "https://uva.fra1.qualtrics.com/CP/Graphic.php?IM=IM_5gKHOiXDNgccIUC",
    "https://uva.fra1.qualtrics.com/CP/Graphic.php?IM=IM_8HMIvQCyobAZL7w",
    "https://uva.fra1.qualtrics.com/CP/Graphic.php?IM=IM_06wcuQzizRBPv4W",
    "https://uva.fra1.qualtrics.com/CP/Graphic.php?IM=IM_5ihESCCOt6EoTCS",
    "https://uva.fra1.qualtrics.com/CP/Graphic.php?IM=IM_85MzZ3DMlTpO3FI",
    "https://uva.fra1.qualtrics.com/CP/Graphic.php?IM=IM_4PgKntGbNS9p7j8",
    "https://uva.fra1.qualtrics.com/CP/Graphic.php?IM=IM_eeMsvdojMZlVE22"
]

# Target folder to save images
folder_name = "downloaded_images_multimodal"
os.makedirs(folder_name, exist_ok=True)

# Download and save each image
for i, url in enumerate(urls, start=1):
    response = requests.get(url)
    if response.status_code == 200:
        image_path = os.path.join(folder_name, f"image_{i}.jpg")
        with open(image_path, "wb") as f:
            f.write(response.content)
        print(f"Saved: {image_path}")
    else:
        print(f"Failed to download image {i} from {url}")
