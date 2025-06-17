import os
import requests

# List of image URLs
urls = ['https://images.gr-assets.com/books/1450349679l/1442782.jpg', 'https://images.gr-assets.com/books/1356574564l/15870096.jpg', 'https://images.gr-assets.com/books/1328872691l/1816515.jpg', 'https://images.gr-assets.com/books/1356135542l/555274.jpg', 'https://images.gr-assets.com/books/1387790017l/1728883.jpg', 'https://images.gr-assets.com/books/1354931244l/15780375.jpg', 'https://images.gr-assets.com/books/1328837581l/6928833.jpg', 'https://images.gr-assets.com/books/1468992944l/30844277.jpg', 'https://images.gr-assets.com/books/1450000591l/25734081.jpg']

# Target folder to save images
folder_name = "downloaded_images_multimodal_good"
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
