import os
import requests
from io import BytesIO
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from django.core.management.base import BaseCommand
from books.models import Book
from tqdm import tqdm
from models.EmotionCLIP.src.models.base import EmotionCLIP

class Command(BaseCommand):
    help = "Compute and store EmotionCLIP embeddings for each book cover."

    def add_arguments(self, parser):
        parser.add_argument(
            "--batch-size", 
            type=int, 
            default=32,
            help="How many books to process in each batch"
        )
        parser.add_argument(
            "--limit", 
            type=int, 
            default=None,
            help="Maximum number of books to process (None for all)"
        )
        parser.add_argument(
            "--skip-existing", 
            action="store_true",
            help="Skip books that already have embeddings"
        )
        parser.add_argument(
            "--timeout", 
            type=int, 
            default=10,
            help="Timeout for image downloads in seconds"
        )

    def handle(self, *args, **options):
        batch_size = options["batch_size"]
        limit = options["limit"]
        skip_existing = options["skip_existing"]
        timeout = options["timeout"]
        
        # 1) Set up preprocessing pipeline based on CLIP requirements
        # These transforms match what CLIP/EmotionCLIP expects
        preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073], 
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        
        # 2) Load EmotionCLIP model
        self.stdout.write("Loading EmotionCLIP model...")
        ckpt_path = "models/EmotionCLIP/emotionclip_latest.pt"
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu")

            # 1) If it’s a Lightning checkpoint dict:
            if "state_dict" in checkpoint:
                hparams = checkpoint.get("hyper_parameters", {})
                model = EmotionCLIP(**hparams)           # re-create with saved args
                state = checkpoint["state_dict"]
            # 2) Or if it’s just a raw state_dict:
            else:
                model = EmotionCLIP()                    # use default init
                state = checkpoint

            model.load_state_dict(state)
            model.eval()
            self.stdout.write(self.style.SUCCESS("Model loaded successfully"))
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Failed to load model: {e}"))
            return
        
        # 3) Get books to process
        filter_kwargs = {}
        if skip_existing and hasattr(Book, 'embedding'):
            filter_kwargs = {"embedding__isnull": True}
            
        books_qs = Book.objects.filter(**filter_kwargs)
        if limit:
            books_qs = books_qs[:limit]
            
        total_books = books_qs.count()
        self.stdout.write(f"Processing {total_books} books...")
        
        # 4) Process books in batches
        processed_count = 0
        failed_count = 0
        
        # Use tqdm for a progress bar
        for book in tqdm(books_qs.iterator(), total=total_books):
            try:
                # Download image
                if not book.image_url:
                    self.stdout.write(f"Book {book.pk} has no image URL, skipping")
                    continue
                
                resp = requests.get(book.image_url_large, timeout=timeout)
                resp.raise_for_status()  # Raise exception for 4xx/5xx errors
                
                # Process image
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                x = preprocess(img).unsqueeze(0)  # Add batch dimension
                
                # Generate embedding
                with torch.no_grad():
                    vec = model.encode_image(x)  # shape [1, D]
                
                # Check if the model has a dimension mismatch
                if hasattr(Book, 'emotionclip_embedding'):
                    # Store embedding (assuming you have a VectorField)
                    book.emotionclip_embedding = vec.squeeze(0).tolist()
                    book.save(update_fields=["emotionclip_embedding"])
                    processed_count += 1
                    
                    if processed_count % 100 == 0:
                        self.stdout.write(f"Processed {processed_count}/{total_books} books")
                else:
                    self.stdout.write(f"Book model doesn't have 'emotionclip_embedding' field. Add a VectorField to Book model.")
                    break
                
            except requests.RequestException as e:
                self.stderr.write(f"Error downloading image for book {book.pk}: {e}")
                failed_count += 1
            except Exception as e:
                self.stderr.write(f"Error processing book {book.pk}: {e}")
                failed_count += 1
        
        # 5) Final report
        self.stdout.write(self.style.SUCCESS(
            f"Finished! Processed {processed_count} books successfully. Failed: {failed_count}"
        ))