import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

async def main():
    # Suppose we want to keep only internal links, remove certain domains, 
    # and discard external images from the final crawl data.
    crawler_cfg = CrawlerRunConfig(
        exclude_external_links=True,
        scan_full_page=True,  # Tells the crawler to try scrolling the entire page
        scroll_delay=0.5,     # Delay (seconds) between scroll steps
        exclude_social_media_links=True,   # skip Twitter, Facebook, etc.
        # exclude_external_images=True,      # keep only images from main domain
        wait_for_images=True,             # ensure images are loaded
        verbose=True
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun("https://aesthetics.fandom.com/wiki/Cottagecore", config=crawler_cfg)

        if result.success:
            print("[OK] Crawled:", result.url)

            # 1. Links
            in_links = result.links.get("internal", [])
            ext_links = result.links.get("external", [])
            print("Internal link count:", len(in_links))
            print("External link count:", len(ext_links))  # should be zero with exclude_external_links=True
            print("Internal links:")
            for i, link in enumerate(in_links):
                if "https://aesthetics.fandom.com/wiki/File:" in link.get('href'):
                    print(f"  - {link['href']} (text={link.get('text','')})")

            # 2. Images
            images = result.media.get("images", [])
            print("Images found:", len(images))

            # Let's see a snippet of these images
            for i, img in enumerate(images[:3]):
                print(f"  - {img['src']} (alt={img.get('alt','')}, score={img.get('score','N/A')})")
        else:
            print("[ERROR] Failed to crawl. Reason:", result.error_message)

if __name__ == "__main__":
    asyncio.run(main())