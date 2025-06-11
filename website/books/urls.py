from django.urls import path
from .views import BookDetailView, ImageSearchView, ImageProcessingView, BookRecommendationsView

urlpatterns = [
    path('', ImageSearchView.as_view(), name='image_search'),
    path('processing/', ImageProcessingView.as_view(), name='image_search_processing'),
    path('recommendations/', BookRecommendationsView.as_view(), name='book_recommendations'),
    path('book/<int:pk>/', BookDetailView.as_view(), name='book_detail'),
]