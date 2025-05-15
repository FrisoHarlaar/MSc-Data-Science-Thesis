from django.urls import path
from .views import BookDetailView, ImageSearchView, ImageProcessingView

urlpatterns = [
    path('book/<int:pk>/', BookDetailView.as_view(), name='book_detail'),
    path('search/by-image/', ImageSearchView.as_view(), name='image_search'),
    path('search/by-image/processing/', ImageProcessingView.as_view(), name='image_search_processing'),
]