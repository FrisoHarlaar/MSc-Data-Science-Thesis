import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from tqdm import tqdm
from django.core.management.base import BaseCommand
from django.db import transaction
from books.models import Book

class Command(BaseCommand):
    help = "Import emotion vectors from parquet files to existing Book records"

    def add_arguments(self, parser):
        parser.add_argument(
            "--resnet-path",
            type=str,
            help="Path to ResNet emotion predictions parquet file"
        )
        parser.add_argument(
            "--bert-path", 
            type=str,
            help="Path to BERT emotion predictions parquet file"
        )
        parser.add_argument(
            "--batch-size", "-b",
            type=int,
            default=5000,
            help="Number of rows per batch"
        )
        parser.add_argument(
            "--skip",
            type=int,
            default=0,
            help="Number of rows to skip before processing"
        )

    def handle(self, *args, **options):
        resnet_path = options.get("resnet_path")
        bert_path = options.get("bert_path")
        batch_size = options["batch_size"]
        skip = options["skip"]
        
        if not resnet_path and not bert_path:
            self.stdout.write(self.style.ERROR(
                "Please provide at least one of --resnet-path or --bert-path"
            ))
            return

        start_time = time.time()
        
        # Process ResNet emotions if provided
        if resnet_path:
            self.stdout.write(f"Processing ResNet emotions from: {resnet_path}")
            self.process_emotion_file(
                path=resnet_path,
                emotion_type="resnet", 
                batch_size=batch_size,
                skip=skip
            )
        
        # Process BERT emotions if provided  
        if bert_path:
            self.stdout.write(f"Processing BERT emotions from: {bert_path}")
            self.process_emotion_file(
                path=bert_path,
                emotion_type="bert",
                batch_size=batch_size, 
                skip=skip
            )
        
        # Compute multimodal combinations if we have both
        if resnet_path and bert_path:
            self.stdout.write("Computing multimodal emotion combinations...")
            self.compute_multimodal_emotions(batch_size=batch_size)
        
        elapsed = time.time() - start_time
        self.stdout.write(self.style.SUCCESS(
            f"Vector import complete in {elapsed:.1f}s"
        ))

    def process_emotion_file(self, path, emotion_type, batch_size, skip):
        """Process a single emotion prediction file"""
        
        # Expected emotion probability columns (based on your notebook)
        emotion_columns = [
            'prob_amusement', 'prob_anger', 'prob_awe', 'prob_contentment',
            'prob_disgust', 'prob_excitement', 'prob_fear', 'prob_sadness', 
            'prob_something_else'
        ]
        
        # Get file metadata
        parquet_file = pq.ParquetFile(path)
        total_rows = parquet_file.metadata.num_rows
        self.stdout.write(f"File has {total_rows} rows for {emotion_type}")
        
        # Set up progress bar
        pbar = tqdm(total=total_rows, unit="rows", desc=f"Processing {emotion_type}")
        
        global_row = 0
        updated_count = 0
        
        for record_batch in parquet_file.iter_batches(batch_size=batch_size):
            batch_len = record_batch.num_rows
            
            # Skip batches if needed
            if global_row + batch_len <= skip:
                global_row += batch_len
                pbar.update(batch_len)
                continue
                
            # Slice off already-done rows
            batch_skip = max(0, skip - global_row)
            batch = record_batch.slice(batch_skip)
            global_row += batch_len
            
            # Convert to pandas
            table = pa.Table.from_batches([batch])
            df = table.to_pandas()
            
            # Extract required columns
            required_columns = ['book_id'] + emotion_columns + ['predicted_emotion', 'confidence']
            available_columns = [col for col in required_columns if col in df.columns]
            
            if 'book_id' not in available_columns:
                self.stdout.write(self.style.ERROR(f"book_id column missing from {emotion_type} file"))
                continue
                
            # Filter to only needed columns
            df_filtered = df[available_columns].copy()
            
            # Remove rows with missing book_id or emotion data
            df_filtered = df_filtered.dropna(subset=['book_id'])
            
            # Prepare update data
            updates = []
            for _, row in df_filtered.iterrows():
                book_id = row['book_id']
                
                # Extract emotion vector (ensure we have all 9 dimensions)
                emotion_vector = []
                for col in emotion_columns:
                    if col in row and pd.notna(row[col]):
                        emotion_vector.append(float(row[col]))
                    else:
                        emotion_vector.append(0.0)
                
                # Skip if vector is all zeros or invalid
                if len(emotion_vector) != 9 or sum(emotion_vector) == 0:
                    continue
                
                update_data = {
                    'book_id': book_id,
                    'emotion_embedding': emotion_vector,
                }
                
                # Add predicted emotion and confidence if available
                if 'predicted_emotion' in row and pd.notna(row['predicted_emotion']):
                    update_data['predicted_emotion'] = row['predicted_emotion']
                    
                if 'confidence' in row and pd.notna(row['confidence']):
                    update_data['confidence'] = float(row['confidence'])
                
                updates.append(update_data)
            
            # Bulk update books
            if updates:
                batch_updated = self.bulk_update_emotions(updates, emotion_type)
                updated_count += batch_updated
            
            pbar.update(batch.num_rows - batch_skip)
        
        pbar.close()
        self.stdout.write(self.style.SUCCESS(
            f"Updated {updated_count} books with {emotion_type} emotions"
        ))

    def bulk_update_emotions(self, updates, emotion_type):
        """Bulk update emotion fields for books"""
        updated_count = 0
        
        # Map emotion type to field names
        field_mapping = {
            'resnet': {
                'embedding': 'artemis_emotion_embedding',
                'emotion': 'artemis_predicted_emotion', 
                'confidence': 'artemis_confidence'
            },
            'bert': {
                'embedding': 'bert_emotion_embedding',
                'emotion': 'bert_predicted_emotion',
                'confidence': 'bert_confidence'
            }
        }
        
        if emotion_type not in field_mapping:
            return 0
            
        fields = field_mapping[emotion_type]
        
        # Group updates by chunks for efficient processing
        chunk_size = 1000
        for i in range(0, len(updates), chunk_size):
            chunk = updates[i:i + chunk_size]
            book_ids = [update['book_id'] for update in chunk]
            
            # Get existing books in one query
            existing_books = Book.objects.filter(book_id__in=book_ids)
            book_lookup = {book.book_id: book for book in existing_books}
            
            # Prepare books for bulk update
            books_to_update = []
            for update in chunk:
                book_id = update['book_id']
                if book_id in book_lookup:
                    book = book_lookup[book_id]
                    
                    # Update emotion embedding
                    setattr(book, fields['embedding'], update['emotion_embedding'])
                    
                    # Update predicted emotion if available
                    if 'predicted_emotion' in update:
                        setattr(book, fields['emotion'], update['predicted_emotion'])
                    
                    # Update confidence if available    
                    if 'confidence' in update:
                        setattr(book, fields['confidence'], update['confidence'])
                    
                    books_to_update.append(book)
            
            # Bulk update
            if books_to_update:
                with transaction.atomic():
                    update_fields = [fields['embedding']]
                    if any('predicted_emotion' in update for update in chunk):
                        update_fields.append(fields['emotion'])
                    if any('confidence' in update for update in chunk):
                        update_fields.append(fields['confidence'])
                        
                    Book.objects.bulk_update(
                        books_to_update, 
                        update_fields,
                        batch_size=500
                    )
                    updated_count += len(books_to_update)
        
        return updated_count

    def compute_multimodal_emotions(self, batch_size=5000):
        """Compute confidence-weighted multimodal emotion combinations"""
        
        # Get books that have both ResNet and BERT embeddings
        books_with_both = Book.objects.filter(
            artemis_emotion_embedding__isnull=False,
            bert_emotion_embedding__isnull=False,
            artemis_confidence__isnull=False,
            bert_confidence__isnull=False
        )
        
        total_books = books_with_both.count()
        self.stdout.write(f"Computing multimodal embeddings for {total_books} books")
        
        if total_books == 0:
            return
            
        pbar = tqdm(total=total_books, unit="books", desc="Computing multimodal")
        updated_count = 0
        
        # Process in batches
        for offset in range(0, total_books, batch_size):
            batch_books = list(books_with_both[offset:offset + batch_size])
            books_to_update = []
            
            for book in batch_books:
                try:
                    # Get embeddings and confidences
                    artemis_embedding = np.array(book.artemis_emotion_embedding)
                    bert_embedding = np.array(book.bert_emotion_embedding)
                    artemis_conf = book.artemis_confidence
                    bert_conf = book.bert_confidence
                    
                    # Compute confidence weights
                    total_conf = artemis_conf + bert_conf
                    if total_conf > 0:
                        artemis_weight = artemis_conf / total_conf
                        bert_weight = bert_conf / total_conf
                        
                        # Weighted combination
                        multimodal_embedding = (
                            artemis_weight * artemis_embedding + 
                            bert_weight * bert_embedding
                        )
                        
                        # Normalize to ensure proper probability distribution
                        multimodal_embedding = multimodal_embedding / np.sum(multimodal_embedding)
                        
                        # Update book
                        book.multimodal_emotion_embedding = multimodal_embedding.tolist()
                        books_to_update.append(book)
                        
                except Exception as e:
                    self.stdout.write(self.style.WARNING(
                        f"Error computing multimodal for book {book.book_id}: {e}"
                    ))
                    continue
            
            # Bulk update
            if books_to_update:
                with transaction.atomic():
                    Book.objects.bulk_update(
                        books_to_update,
                        ['multimodal_emotion_embedding'],
                        batch_size=500
                    )
                    updated_count += len(books_to_update)
            
            pbar.update(len(batch_books))
        
        pbar.close()
        self.stdout.write(self.style.SUCCESS(
            f"Computed multimodal embeddings for {updated_count} books"
        ))