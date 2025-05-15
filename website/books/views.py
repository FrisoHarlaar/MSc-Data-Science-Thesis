import torch
from io import BytesIO
from PIL import Image
from torchvision import transforms
from models.EmotionCLIP.src.models.base import EmotionCLIP

from django.shortcuts import render
from django.views.generic import DetailView, FormView
from django.db.models import Q, Case, When, IntegerField
from django.utils.text import slugify

from .models import Book
from .forms import ImageUploadForm
from .services import ImageEmbeddingService


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
            # The shelves are already in the correct format
            formatted_shelves = book.popular_shelves
        else:
            formatted_shelves = []
            
        context['formatted_shelves'] = formatted_shelves
        
        # If we have the emotionclip embedding, we could use it for recommendations
        if hasattr(book, 'emotionclip_embedding') and book.emotionclip_embedding:
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
            # Return empty queryset if no similar books or invalid format
            return Book.objects.none()
        
        # Get the list of book IDs
        similar_book_ids = book.similar_books
        
        if not similar_book_ids:
            return Book.objects.none()
            
        # Limit to first 10 IDs to avoid excessive queries
        similar_book_ids = similar_book_ids[:10]
        
        # Query books with those IDs
        similar_books = Book.objects.filter(book_id__in=similar_book_ids)
        
        # If we got no matches, just return the empty queryset
        if not similar_books.exists():
            return similar_books
            
        # Sort the books according to their order in the similar_books list
        # This preserves the original ranking/relevance
        preserved_order = {book_id: i for i, book_id in enumerate(similar_book_ids)}
        
        # Use a custom ordering using Case/When
        order_cases = [
            When(book_id=book_id, then=pos) for book_id, pos in preserved_order.items()
        ]
        
        if order_cases:
            similar_books = similar_books.annotate(
                position=Case(*order_cases, output_field=IntegerField())
            ).order_by('position')
            
        # Exclude the current book (just in case it's in its own similar_books list)
        similar_books = similar_books.exclude(pk=book.pk)
        
        return similar_books
    

from django.views.generic import FormView
from django.shortcuts import render, redirect
from .forms import ImageUploadForm
from .models import Book, UserImage
from django.urls import reverse
from .services import get_embedding_service
import uuid
import logging
from django.db.models import Q
from pgvector.django import L2Distance
import io

logger = logging.getLogger(__name__)

class ImageSearchView(FormView):
    template_name = 'books/image_search.html'
    form_class = ImageUploadForm
    success_url = '#results'
    
    def form_valid(self, form):
        # Get the uploaded image
        uploaded_image = form.cleaned_data['image']
        limit = form.cleaned_data.get('limit', 6)
        emotion_filter = form.cleaned_data.get('emotion_filter', None)
        
        try:
            # Read the image data
            image_data = uploaded_image.read()
            
            # Create a UserImage entry to save this upload
            user_image = UserImage(image=uploaded_image)
            user_image.save()
            
            # Get the embedding service and submit image for processing
            embedding_service = get_embedding_service()
            request_id = embedding_service.get_embedding_async(image_data)
            
            # Redirect to processing page with the request ID and image ID
            return redirect(reverse('image_search_processing') + 
                           f'?request_id={request_id}&image_id={user_image.uuid}&limit={limit}' + 
                           (f'&emotion_filter={emotion_filter}' if emotion_filter else ''))
            
        except Exception as e:
            logger.exception("Error processing image upload")
            form.add_error(None, f"Error processing image: {e}")
            return self.form_invalid(form)


class ImageProcessingView(FormView):
    """View to show processing status and poll for results"""
    template_name = 'books/image_processing.html'
    form_class = ImageUploadForm  # Reuse the same form for consistency
    
    def get(self, request, *args, **kwargs):
        # Get parameters
        request_id = request.GET.get('request_id')
        image_id = request.GET.get('image_id')
        limit = int(request.GET.get('limit', 6))
        emotion_filter = request.GET.get('emotion_filter')
        
        if not request_id or not image_id:
            return redirect('image_search')
        
        try:
            # Get the user uploaded image
            user_image = UserImage.objects.get(uuid=image_id)
            
            # Check if we have a result already
            embedding_service = get_embedding_service()
            result = embedding_service.get_embedding_result(request_id, timeout=1)
            
            if result['status'] == 'success':
                # We have an embedding! Get similar books
                embedding = result['embedding']
                
                # Save embedding to user image
                user_image.embedding = embedding
                user_image.save()
                
                # Find similar books
                similar_books = self.find_similar_books(embedding, limit, emotion_filter)
                
                # Show results page
                context = self.get_context_data()
                context['similar_books'] = similar_books
                context['user_image'] = user_image
                context['submitted'] = True
                context['processing_complete'] = True
                
                return render(request, 'books/image_search_results.html', context)
            else:
                # Still processing - show loading page
                context = self.get_context_data()
                context['request_id'] = request_id
                context['image_id'] = image_id
                context['limit'] = limit
                context['emotion_filter'] = emotion_filter
                context['user_image'] = user_image
                
                if result.get('error'):
                    context['error'] = result['error']
                    
                return render(request, 'books/image_processing.html', context)
                
        except UserImage.DoesNotExist:
            return redirect('image_search')
        except Exception as e:
            logger.exception(f"Error in image processing view: {e}")
            return redirect('image_search')
    
    def find_similar_books(self, embedding, limit=6, emotion_filter=None):
        """Find books with similar embeddings"""
        # Base query - only consider books with embeddings
        query = Book.objects.filter(emotionclip_embedding__isnull=False)
        
        # Apply emotion filter if provided
        if emotion_filter and emotion_filter != 'all':
            # Filter based on popular_shelves
            if isinstance(emotion_filter, str):
                query = query.filter(popular_shelves__contains=[{"name": emotion_filter}])
        
        # Order by vector similarity
        similar_books = query.order_by(
            L2Distance('emotionclip_embedding', embedding)
        )[:limit]
        
        return similar_books