# your_app/management/commands/import_books.py

import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from django.core.management.base import BaseCommand
from django.db import transaction
from books.models import Book

def ensure_list(val):
    """
    If val has a .tolist() method (e.g. numpy.ndarray or pyarrow ListArray),
    call it so we get plain Python lists. Otherwise return val unchanged.
    """
    if hasattr(val, "tolist"):
        try:
            return val.tolist()
        except Exception:
            pass
    return val

class Command(BaseCommand):
    help = "Stream-import books from a Parquet file in fixed-size batches (no OOM)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--path", "-p",
            type=str,
            required=True,
            help="Path to the Parquet file"
        )
        parser.add_argument(
            "--batch-size", "-b",
            type=int,
            default=10_000,
            help="Number of rows per batch"
        )
        parser.add_argument("--skip",
            type=int,
            default=0,
            help="Number of rows to skip before inserting (for resuming)"
        )


    def handle(self, *args, **options):
        path = options["path"]
        batch_size = options["batch_size"]

        # --- metadata only, so we know total rows for the progress bar ---
        parquet_file = pq.ParquetFile(path)
        total_rows = parquet_file.metadata.num_rows
        self.stdout.write(f"File has {total_rows} rows.")

        # --- set up timer & progress bar ---
        start_time = time.time()
        pbar = tqdm(total=total_rows, unit="rows", desc="Importing")

        offset = options["skip"]
        global_row = 0

        for record_batch in parquet_file.iter_batches(batch_size=batch_size):
            batch_len = record_batch.num_rows

            # If we've not yet reached the offset, skip whole batches
            if global_row + batch_len <= offset:
                global_row += batch_len
                pbar.update(batch_len)
                continue

            # Otherwise, slice off the already-done rows within this batch
            skip = max(0, offset - global_row)
            batch = record_batch.slice(skip)
            global_row += batch_len

            # Convert & insert `batch` as before…
            table = pa.Table.from_batches([batch])
            df = table.to_pandas().where(pd.notnull(table.to_pandas()), None)

            # 2) Force any list/array types into Python lists
            for col in ("authors", "popular_shelves", "similar_books"):
                if col in df.columns:
                    df[col] = df[col].apply(ensure_list)

            # build your Book instances
            books = []
            for rec in df.to_dict(orient="records"):
                if rec.get('title') is None:
                    self.stdout.write(self.style.WARNING(
                        f"Skipping row {rec.get('book_id')} with no title."
                    ))
                    continue
                books.append(Book(
                    book_id=rec.get('book_id'),
                    isbn13=rec.get("isbn13"),
                    isbn=rec.get("isbn"),
                    title=rec.get("title"),
                    title_without_series=rec.get("title_without_series"),
                    series=rec.get("series"),
                    authors=rec.get("authors"),
                    description=rec.get("description"),
                    publisher=rec.get("publisher"),
                    publication_year=rec.get("publication_year"),
                    publication_month=rec.get("publication_month"),
                    publication_day=rec.get("publication_day"),
                    edition_information=rec.get("edition_information"),
                    format=rec.get("format"),
                    num_pages=rec.get("num_pages"),
                    language_code=rec.get("language_code"),
                    country_code=rec.get("country_code"),
                    url=rec.get("url"),
                    link=rec.get("link"),
                    image_url=rec.get("image_url"),
                    image_url_large=rec.get("image_url_large"),
                    average_rating=rec.get("average_rating"),
                    ratings_count=rec.get("ratings_count"),
                    text_reviews_count=rec.get("text_reviews_count"),
                    popular_shelves=rec.get("popular_shelves"),
                    similar_books=rec.get("similar_books"),
                ))

            # bulk-insert this batch
            with transaction.atomic():
                Book.objects.bulk_create(books, ignore_conflicts=True)

            # update progress bar
            pbar.update(batch.num_rows - skip)

        pbar.close()
        elapsed = time.time() - start_time
        self.stdout.write(self.style.SUCCESS(
            f"Import complete: {total_rows} rows in {elapsed:.1f}s "
            f"({total_rows/elapsed:.0f} rows/s)."
        ))
