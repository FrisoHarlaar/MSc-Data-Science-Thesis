import torch
from io import BytesIO
from PIL import Image
from django.shortcuts import render, redirect
from django.views.generic import DetailView, FormView
from django.db.models import Q, Case, When, IntegerField
from django.utils.text import slugify
from django.urls import reverse
from pgvector.django import L2Distance, CosineDistance
import numpy as np
import logging

from .models import Book, UserImage
from .forms import ImageUploadForm
from .services.image_emotion_service import get_emotion_service

logger = logging.getLogger(__name__)

class BookDetailView(DetailView):
    model = Book
    template_name = 'books/book_detail.html'
    context_object_name = 'book'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['similar_books'] = self.get_similar_books()
        
        # Process the popular shelves data
        book = self.object
        if book.popular_shelves and isinstance(book.popular_shelves, list):
            formatted_shelves = book.popular_shelves
        else:
            formatted_shelves = []
            
        context['formatted_shelves'] = formatted_shelves
        
        # Check if we have emotion embeddings for visual recommendations
        if (hasattr(book, 'artemis_emotion_embedding') and book.artemis_emotion_embedding):
            context['has_visual_recommendations'] = True
        else:
            context['has_visual_recommendations'] = False
            
        return context

    def get_similar_books(self):
        """
        Fetch similar books based on the book_ids stored in the similar_books field.
        Returns a queryset of Book objects ordered by their position in the similar_books list.
        """
        book = self.object
        
        # Check if the book has similar_books data
        if not book.similar_books or not isinstance(book.similar_books, list):
            return Book.objects.none()
        
        # Get the list of book IDs
        similar_book_ids = book.similar_books[:10]  # Limit to first 10
        
        if not similar_book_ids:
            return Book.objects.none()
            
        # Query books with those IDs
        similar_books = Book.objects.filter(book_id__in=similar_book_ids)
        
        if not similar_books.exists():
            return similar_books
            
        # Sort the books according to their order in the similar_books list
        preserved_order = {book_id: i for i, book_id in enumerate(similar_book_ids)}
        
        # Use custom ordering using Case/When
        order_cases = [
            When(book_id=book_id, then=pos) for book_id, pos in preserved_order.items()
        ]
        
        if order_cases:
            similar_books = similar_books.annotate(
                position=Case(*order_cases, output_field=IntegerField())
            ).order_by('position')
            
        # Exclude the current book
        similar_books = similar_books.exclude(pk=book.pk)
        
        return similar_books


class ImageSearchView(FormView):
    template_name = 'books/image_search.html'
    form_class = ImageUploadForm
    success_url = '#results'
    
    def form_valid(self, form):
        # Get the uploaded image
        uploaded_image = form.cleaned_data['image']
        limit = form.cleaned_data.get('limit', 6)
        approach = form.cleaned_data.get('approach', 'multimodal')
        
        try:
            # Read the image data
            image_data = uploaded_image.read()
            
            # Create a UserImage entry to save this upload
            user_image = UserImage(image=uploaded_image)
            user_image.save()
            
            # Get the emotion service and submit image for processing
            emotion_service = get_emotion_service()
            request_id = emotion_service.get_emotion_async(image_data)
            
            # Redirect to processing page with the request ID and image ID
            return redirect(reverse('image_search_processing') + 
                           f'?request_id={request_id}&image_id={user_image.uuid}&limit={limit}&approach={approach}')
            
        except Exception as e:
            logger.exception("Error processing image upload")
            form.add_error(None, f"Error processing image: {e}")
            return self.form_invalid(form)


class ImageProcessingView(FormView):
    """View to show processing status and poll for results"""
    template_name = 'books/image_processing.html'
    form_class = ImageUploadForm
    
    def get(self, request, *args, **kwargs):
        # Get parameters
        request_id = request.GET.get('request_id')
        image_id = request.GET.get('image_id')
        limit = int(request.GET.get('limit', 6))
        approach = request.GET.get('approach', 'multimodal')
        
        if not request_id or not image_id:
            return redirect('image_search')
        
        try:
            # Get the user uploaded image
            user_image = UserImage.objects.get(uuid=image_id)
            
            # Check if we have a result already
            emotion_service = get_emotion_service()
            result = emotion_service.get_emotion_result(request_id, timeout=1)
            
            if result['status'] == 'success':
                # We have emotions! Find similar books
                emotion_distribution = result['emotion_distribution']
                
                # Save emotion data to user image for reference
                user_image.artemis_emotion_embedding = emotion_distribution.tolist()
                user_image.save()
                
                # Find similar books using the selected approach
                similar_books = self.find_similar_books_by_emotion(
                    emotion_distribution, limit, approach
                )
                
                # Show results page
                context = self.get_context_data()
                context['similar_books'] = similar_books
                context['user_image'] = user_image
                context['emotion_prediction'] = result
                context['approach'] = approach
                context['submitted'] = True
                context['processing_complete'] = True
                
                return render(request, 'books/image_search_results.html', context)
            else:
                # Still processing or error - show loading/error page
                context = self.get_context_data()
                context['request_id'] = request_id
                context['image_id'] = image_id
                context['limit'] = limit
                context['approach'] = approach
                context['user_image'] = user_image
                
                if result.get('error'):
                    context['error'] = result['error']
                    
                return render(request, 'books/image_processing.html', context)
                
        except UserImage.DoesNotExist:
            return redirect('image_search')
        except Exception as e:
            logger.exception(f"Error in image processing view: {e}")
            return redirect('image_search')
    
    def find_similar_books_by_emotion(self, emotion_distribution, limit=6, approach='multimodal'):
        """Find books with similar emotion distributions using vector similarity"""
        
        # Choose the appropriate embedding field based on approach
        if approach == 'artemis':
            embedding_field = 'artemis_emotion_embedding'
        elif approach == 'bert':
            embedding_field = 'bert_emotion_embedding'
        else:  # multimodal
            embedding_field = 'multimodal_emotion_embedding'
        
        # Base query - only consider books with embeddings
        query = Book.objects.filter(**{f'{embedding_field}__isnull': False})
        
        # Convert emotion distribution to list for pgvector
        if isinstance(emotion_distribution, np.ndarray):
            emotion_list = emotion_distribution.tolist()
        else:
            emotion_list = emotion_distribution
        
        # Order by vector similarity using cosine distance (smaller is more similar)
        similar_books = query.order_by(
            CosineDistance(embedding_field, emotion_list)
        )[:limit]
        
        return similar_books