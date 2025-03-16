import asyncio
import os
import json
import urllib.parse
import re
import requests
import glob
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, LXMLWebScrapingStrategy
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy


# Problems:
# Fashwave is part of PoliticalWave now, has no gallery
# Frogcore is a subaesthetic, has no gallery

AESTHETICS = [
    # "Anglo_Gothic",
    # "Angura_Kei",
    # "Atompunk",
    # "Bloomcore",
    # "Bubblegum_Witch",
    # "Cottagecore",
    # "Cyberpunk",
    # "Dark_Academia",
    # "Ethereal",
    # "Fairy_Kei",
    # "Fashwave",
    # "Frogcore",
    # "Goblincore",
    # "Grandparentcore",
    # "Hellenic",
    # "Kidcore",
    # "Light_Academia",
    # "Monkeycore",
    # "Pastel_Goth",
    # "Princesscore",
    # "Traumacore",
    "Vaporwave",
    "VaporGoth",
    "Vibrant_Academia",
    "Virgo%27s_Tears",
]

def download_image(image_url, save_path):
    """Download an image from image_url and save it to save_path."""
    # Skip if file already exists
    if os.path.exists(save_path):
        print(f"File already exists: {save_path}")
        return True
        
    # Continue with existing download logic
    response = requests.get(image_url)
    if response.status_code == 200:
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(save_path)
            os.makedirs(directory, exist_ok=True)
            
            # Write the file
            with open(save_path, "wb") as f:
                f.write(response.content)
            return True
        except (OSError, IOError) as e:
            print(f"Error saving {save_path}: {e}")
            
            # Try with a sanitized filename
            try:
                directory = os.path.dirname(save_path)
                filename = os.path.basename(save_path)
                
                # Decode URL-encoded characters
                decoded_filename = urllib.parse.unquote(filename)
                
                # Replace problematic characters with underscores
                sanitized_filename = re.sub(r'[\\/*?:"<>|%]', '_', decoded_filename)
                
                # Limit filename length (Windows has 260 character path limit)
                if len(sanitized_filename) > 100:
                    name_part, ext_part = os.path.splitext(sanitized_filename)
                    sanitized_filename = name_part[:95] + ext_part
                    
                new_save_path = os.path.join(directory, sanitized_filename)
                # Check if sanitized file already exists
                if os.path.exists(new_save_path):
                    print(f"Sanitized file already exists: {new_save_path}")
                    return True
                    
                with open(new_save_path, "wb") as f:
                    f.write(response.content)
                print(f"Saved with sanitized filename: {sanitized_filename}")
                return True
            except Exception as e2:
                print(f"Failed to save even with sanitized filename: {e2}")
                return False
    else:
        print(f"Failed to download {image_url}")
        return False

def get_full_res_url(downscaled_url):
    """
    Remove the '/revision' portion from the URL.
    For example, converts:
    https://.../53267148_n.jpg/revision/latest/scale-to-width-down/130?cb=...
    into:
    https://.../53267148_n.jpg
    """
    if '/revision' in downscaled_url:
        return downscaled_url.split('/revision')[0]
    return downscaled_url

async def main():
    # Define a JSON extraction schema that selects all <a class="image lightbox">
    # elements and extracts the src attribute from their nested <img> tag.
    schema = {
        "name": "GalleryImages",
        "baseSelector": "a.image.lightbox",
        "fields": [
            {
                "name": "img_src",
                "selector": "img",
                "type": "attribute",
                "attribute": "src"
            }
        ]
    }
    extraction_strategy = JsonCssExtractionStrategy(schema)
    
    # Configure the crawler using crawl4ai's built-in CSS selection.
    config = CrawlerRunConfig(
        css_selector="a.image.lightbox",         # limit selection to lightbox anchors
        extraction_strategy=extraction_strategy,  # use our JSON extraction schema
        exclude_external_links=True,
        wait_for_images=True,
        scan_full_page=True,
        scroll_delay=0.5,
        cache_mode=CacheMode.BYPASS,
        scraping_strategy=LXMLWebScrapingStrategy()  # for improved performance
    )

    for aesthetic_name in AESTHETICS:
        # Configure folder path
        folder = os.path.join("data", "aesthetic_images", aesthetic_name)
        
        # Check if folder exists
        if os.path.exists(folder):
            # Count files in the folder
            existing_files = glob.glob(os.path.join(folder, "*"))
            
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(f"https://aesthetics.fandom.com/wiki/{aesthetic_name}", config=config)
                if not result.success:
                    print(f"Failed to crawl {aesthetic_name}:", result.error_message)
                    continue

                # Parse the JSON extracted content
                data = json.loads(result.extracted_content)
                print(f"{aesthetic_name}: Extracted {len(data)} images, folder has {len(existing_files)} files")
                
                # Skip if we already have all the files
                if len(existing_files) >= len(data):
                    print(f"Folder {folder} already contains all {len(data)} images, skipping...")
                    continue
                
                # Process only missing files
                for item in data:
                    downscaled_url = item.get("img_src", "")
                    if not downscaled_url:
                        continue
                    full_res_url = get_full_res_url(downscaled_url)
                    filename = os.path.basename(full_res_url.split("?")[0])
                    save_path = os.path.join(folder, filename)
                    download_image(full_res_url, save_path)
        else:
            # Folder doesn't exist, process normally
            os.makedirs(folder, exist_ok=True)
            
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(f"https://aesthetics.fandom.com/wiki/{aesthetic_name}", config=config)
                if not result.success:
                    print(f"Failed to crawl {aesthetic_name}:", result.error_message)
                    continue

                # Parse the JSON extracted content
                data = json.loads(result.extracted_content)
                print(f"{aesthetic_name}: Extracted {len(data)} images")
                
                for item in data:
                    downscaled_url = item.get("img_src", "")
                    if not downscaled_url:
                        continue
                    full_res_url = get_full_res_url(downscaled_url)
                    filename = os.path.basename(full_res_url.split("?")[0])
                    save_path = os.path.join(folder, filename)
                    download_image(full_res_url, save_path)

if __name__ == "__main__":
    asyncio.run(main())
