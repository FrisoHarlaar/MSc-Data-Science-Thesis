from django.db import models
from wagtail.admin.panels import FieldPanel
from pgvector.django import VectorField 
import uuid


class Book(models.Model):
    # IDs and unique identifiers
    book_id = models.CharField(max_length=20, unique=True)
    work_id = models.CharField(max_length=20, null=True, blank=True)
    isbn = models.CharField(max_length=255, null=True, blank=True)
    isbn13 = models.CharField(max_length=255, null=True, blank=True)
    asin = models.CharField(max_length=20, null=True, blank=True)
    kindle_asin = models.CharField(max_length=20, null=True, blank=True)
    
    # Basic book information
    title = models.CharField(max_length=255)
    title_without_series = models.CharField(max_length=255, null=True, blank=True)
    series = models.CharField(max_length=255, null=True, blank=True)
    authors = models.JSONField(null=True, blank=True)  # Store authors as JSON
    description = models.TextField(null=True, blank=True)
    
    # Publication info
    publisher = models.CharField(max_length=255, null=True, blank=True)
    publication_year = models.IntegerField(null=True, blank=True)
    publication_month = models.IntegerField(null=True, blank=True)
    publication_day = models.IntegerField(null=True, blank=True)
    edition_information = models.CharField(max_length=255, null=True, blank=True)
    
    # Book details
    format = models.CharField(max_length=100, null=True, blank=True)
    num_pages = models.PositiveIntegerField(null=True, blank=True)
    is_ebook = models.BooleanField(default=False)
    
    # Language and location
    language_code = models.CharField(max_length=10, null=True, blank=True)
    country_code = models.CharField(max_length=10, null=True, blank=True)
    
    # URLs and images
    url = models.TextField(null=True, blank=True)
    link = models.TextField(null=True, blank=True)
    image_url = models.TextField()
    image_url_large = models.TextField(null=True, blank=True)
    
    # Ratings and reviews
    average_rating = models.DecimalField(max_digits=3, decimal_places=2, null=True, blank=True)
    ratings_count = models.PositiveIntegerField(null=True, blank=True)
    text_reviews_count = models.PositiveIntegerField(null=True, blank=True)
    
    # Additional fields that could be added as needed
    popular_shelves = models.JSONField(null=True, blank=True)  # Store shelves as JSON
    similar_books = models.JSONField(null=True, blank=True)  # Store similar book IDs as JSON

    emotionclip_embedding = VectorField(dimensions=512, null=True, blank=True)

    # Artemis emotion predictions (9 emotions)
    artemis_emotion_embedding = VectorField(dimensions=9, null=True, blank=True)
    artemis_predicted_emotion = models.CharField(max_length=50, null=True, blank=True)
    artemis_confidence = models.FloatField(null=True, blank=True)
    
    # BERT emotion predictions (9 emotions) 
    bert_emotion_embedding = VectorField(dimensions=9, null=True, blank=True)
    bert_predicted_emotion = models.CharField(max_length=50, null=True, blank=True)
    bert_confidence = models.FloatField(null=True, blank=True)

    multimodal_emotion_embedding = VectorField(dimensions=9, null=True, blank=True)


    panels = [
        FieldPanel("book_id"),
        FieldPanel("isbn13"),
        FieldPanel("isbn"),
        FieldPanel("title"),
        FieldPanel("authors"),
        FieldPanel("similar_books"),
        FieldPanel("description"),
        FieldPanel("image_url"),
        FieldPanel("image_url_large"),
        FieldPanel("publisher"),
        FieldPanel("publication_year"),
        FieldPanel("format"),
        FieldPanel("num_pages"),
        FieldPanel("language_code"),
        FieldPanel("average_rating"),
        FieldPanel("ratings_count"),
    ]

    def __str__(self):
        return self.title

    class Meta:
        indexes = [
            # Add a pgvector index for faster similarity searches
            models.Index(fields=['emotionclip_embedding'], name='emotionclip_idx'),
            models.Index(fields=['artemis_emotion_embedding'], name='artemis_emotion_idx'),
            models.Index(fields=['bert_emotion_embedding'], name='bert_emotion_idx'),
            models.Index(fields=['multimodal_emotion_embedding'], name='multimodal_emotion_idx'),
        ]


class UserImage(models.Model):
    """Model to store user uploaded images for emotion analysis"""
    image = models.ImageField(upload_to='user_images/%Y/%m/%d/')
    artemis_emotion_embedding = VectorField(dimensions=9, null=True, blank=True)
    dominant_emotion = models.CharField(max_length=50, null=True, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['artemis_emotion_embedding'], name='user_image_emotion_idx'),
        ]