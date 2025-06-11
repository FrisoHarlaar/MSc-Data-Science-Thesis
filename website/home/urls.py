from django.urls import path
from . import views

app_name = 'home'  # Changed from 'users' to 'home'

urlpatterns = [
    path('', views.LandingPageView.as_view(), name='index'),
    # You can also add an alias for 'landing_page' if needed elsewhere
    path('landing/', views.LandingPageView.as_view(), name='landing_page'),
]