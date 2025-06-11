import torch
from io import BytesIO
from PIL import Image
from django.shortcuts import render, redirect
from django.views.generic import DetailView, FormView, TemplateView
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

# Log GPU status at startup
if torch.cuda.is_available():
    logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA version: {torch.version.cuda}")
else:
    logger.warning("GPU not available - using CPU")

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
        if len(book.artemis_emotion_embedding) > 0:
            context['has_visual_recommendations'] = True
            # Try to extract dominant emotion if available
            context['dominant_emotion'] = self.get_dominant_emotion(book)
        else:
            context['has_visual_recommendations'] = False
            
        return context

    def get_dominant_emotion(self, book):
        """
        Extract the dominant emotion from the artemis_emotion_embedding if available
        """
        try:
            if len(book.artemis_emotion_embedding) > 0:
                emotion_labels = [
                    'amusement', 'anger', 'awe', 'contentment', 'disgust',
                    'excitement', 'fear', 'sadness'
                ]
                
                # Convert to numpy array and find max
                emotions = np.array(book.artemis_emotion_embedding)
                if len(emotions) >= len(emotion_labels):
                    dominant_idx = np.argmax(emotions[:len(emotion_labels)])
                    return emotion_labels[dominant_idx]
        except Exception as e:
            logger.warning(f"Error extracting dominant emotion: {e}")
        
        return None

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
        similar_book_ids = book.similar_books[:10]
        
        # Create a case statement to preserve the order
        preserved_order = Case(
            *[When(book_id=book_id, then=pos) for pos, book_id in enumerate(similar_book_ids)],
            output_field=IntegerField()
        )
        
        # Fetch and order the books
        similar_books = Book.objects.filter(
            book_id__in=similar_book_ids
        ).annotate(
            order=preserved_order
        ).order_by('order')
        
        return similar_books

class ImageSearchView(FormView):
    template_name = 'books/image_upload.html'
    form_class = ImageUploadForm
    success_url = '#results'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Add any additional context needed for the template
        context['gpu_available'] = torch.cuda.is_available()
        return context
    
    def form_valid(self, form):
        # Get the uploaded image with default values
        uploaded_image = form.cleaned_data['image']
        limit = 8  

        try:
            # Log the upload details
            logger.info(f"Processing upload: {uploaded_image.name}, size: {uploaded_image.size}")
            
            # Log GPU memory usage
            if torch.cuda.is_available():
                logger.info(f"GPU memory before processing: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            
            # Read the image data
            image_data = uploaded_image.read()
            logger.info(f"Image data read successfully, length: {len(image_data)}")
            
            # Create a UserImage entry to save this upload
            user_image = UserImage(image=uploaded_image)
            user_image.save()
            logger.info(f"UserImage saved with UUID: {user_image.uuid}")
            
            # Get the emotion service and submit image for processing
            emotion_service = get_emotion_service()
            logger.info("Emotion service obtained")
            
            request_id = emotion_service.get_emotion_async(image_data)
            logger.info(f"Async processing started with request_id: {request_id}")
            
            # Redirect to processing page with the request ID and image ID
            redirect_url = reverse('image_search_processing') + f'?request_id={request_id}&image_id={user_image.uuid}&limit={limit}'
            logger.info(f"Redirecting to: {redirect_url}")
            
            return redirect(redirect_url)
            
        except Exception as e:
            logger.exception("Error processing image upload")
            form.add_error(None, f"Error processing image: {e}")
            return self.form_invalid(form)


class ImageProcessingView(FormView):
    """View to show processing status and poll for results"""
    template_name = 'books/loading_screen.html'
    form_class = ImageUploadForm
    
    def get(self, request, *args, **kwargs):
        request_id = request.GET.get('request_id')
        image_id = request.GET.get('image_id')
        limit = int(request.GET.get('limit', 6))
        
        if not request_id or not image_id:
            logger.warning("Missing request_id or image_id in processing view")
            return redirect('image_search')
        
        try:
            user_image = UserImage.objects.get(uuid=image_id)
            logger.info(f"Processing image for request_id: {request_id}, image_id: {image_id}")
            
            # Get emotion service and check if processing is complete
            emotion_service = get_emotion_service()
            
            # First check with very short timeout (almost immediate)
            result = emotion_service.get_emotion_result(request_id, timeout=0.5)
            
            logger.info(f"Emotion service result: {result}")
            
            if result and not result.get('processing', False):
                # Processing already complete - skip loading screen
                if result.get('status') == 'error':
                    logger.error(f"Emotion processing error: {result.get('error')}")
                    context = self.get_context_data()
                    context['error'] = result.get('error', 'Unknown error occurred')
                    context['user_image'] = user_image
                    return render(request, self.template_name, context)
                
                if 'emotion_distribution' not in result:
                    logger.error(f"No emotion_distribution in result: {result}")
                    context = self.get_context_data()
                    context['error'] = 'Failed to analyze image emotions'
                    context['user_image'] = user_image
                    return render(request, self.template_name, context)
                
                # Processing complete - redirect immediately to recommendations
                logger.info("Processing complete, skipping loading screen")
                emotion_distribution = result['emotion_distribution']
                
                # Store the emotion result in session
                request.session['emotion_result'] = result
                request.session['image_id'] = str(image_id)
                request.session['limit'] = limit
                
                # Skip loading screen and go directly to recommendations
                return redirect('book_recommendations')
            else:
                # Still processing - show loading page with auto-refresh
                logger.info("Still processing, showing loading screen")
                context = self.get_context_data()
                context['request_id'] = request_id
                context['image_id'] = image_id
                context['limit'] = limit
                context['user_image'] = user_image
                
                if result and result.get('status') == 'error':
                    context['error'] = result.get('error')
                    
                return render(request, self.template_name, context)
                
        except UserImage.DoesNotExist:
            logger.error(f"UserImage not found: {image_id}")
            return redirect('image_search')
        except Exception as e:
            logger.exception(f"Error in image processing view: {e}")
            return redirect('image_search')

class BookRecommendationsView(TemplateView):
    """View to display book recommendations based on emotion analysis"""
    template_name = 'books/recommendations.html'
    
    def get(self, request, *args, **kwargs):
        # Get the emotion result from session
        emotion_result = request.session.get('emotion_result')
        image_id = request.session.get('image_id')
        limit = request.session.get('limit', 6)
        
        if not emotion_result or not image_id:
            logger.warning("No emotion result or image_id in session, redirecting to image search")
            return redirect('image_search')
        
        try:
            user_image = UserImage.objects.get(uuid=image_id)
            
            # Get emotion distribution
            emotion_distribution = np.array(emotion_result['emotion_distribution'])
            logger.info(f"Displaying recommendations for emotion distribution: {emotion_distribution}")
            
            # Find similar books using emotion similarity
            similar_books = self.find_similar_books_by_emotion(emotion_distribution, limit)
            
            context = self.get_context_data()
            context['similar_books'] = similar_books
            context['user_image'] = user_image
            context['emotion_prediction'] = emotion_result
            context['submitted'] = True
            context['processing_complete'] = True
            
            # Clear the session data since we've used it
            if 'emotion_result' in request.session:
                del request.session['emotion_result']
            if 'image_id' in request.session:
                del request.session['image_id']
            if 'limit' in request.session:
                del request.session['limit']
            
            return render(request, self.template_name, context)
            
        except UserImage.DoesNotExist:
            logger.error(f"UserImage not found: {image_id}")
            return redirect('image_search')
        except Exception as e:
            logger.exception(f"Error in recommendations view: {e}")
            return redirect('image_search')
    
    def find_similar_books_by_emotion(self, emotion_distribution, limit=6):
        """Find books similar to the given emotion distribution"""
        try:
            # Convert emotion_distribution to numpy if it's a tensor
            if torch.is_tensor(emotion_distribution):
                emotion_distribution = emotion_distribution.cpu().numpy()
            elif isinstance(emotion_distribution, list):
                emotion_distribution = np.array(emotion_distribution)
            
            logger.info(f"Finding similar books with limit {limit}")
            
            # Find books with emotion embeddings
            emotion_books = Book.objects.filter(
                artemis_emotion_embedding__isnull=False
            )
            
            if emotion_books.exists():
                # Use pgvector for efficient similarity search
                emotion_books = emotion_books.annotate(
                    emotion_distance=CosineDistance('artemis_emotion_embedding', emotion_distribution.tolist())
                ).order_by('emotion_distance')[:limit * 2]
                
                result_books = emotion_books[:limit]
                logger.info(f"Found {len(result_books)} books using emotion similarity")
                return result_books
            else:
                logger.warning("No books with emotion embeddings found")
                # Fallback to highly-rated books if no emotion embeddings
                return Book.objects.filter(average_rating__gte=4.0).order_by('?')[:limit]
                
        except Exception as e:
            logger.exception(f"Error finding similar books: {e}")
            # Return highly-rated books as fallback
            return Book.objects.filter(average_rating__gte=4.0).order_by('?')[:limit]