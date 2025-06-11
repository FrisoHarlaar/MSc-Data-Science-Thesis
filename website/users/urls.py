from django.urls import path
from . import views

app_name = 'users'

urlpatterns = [
    # Class-based views
    path('profile/', views.UserProfileView.as_view(), name='profile'),
    path('settings/', views.UserSettingsView.as_view(), name='settings'),
    path('favorites/', views.UserFavoritesView.as_view(), name='favorites'),
    path('account/', views.UserAccountView.as_view(), name='account'),
    
    # AJAX endpoints
    path('update-profile/', views.update_profile_ajax, name='update_profile_ajax'),
]