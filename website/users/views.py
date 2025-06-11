from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.models import User
from django.contrib import messages
from django.views.generic import DetailView, UpdateView, TemplateView
from django.urls import reverse_lazy
from django.http import JsonResponse

from .models import UserProfile
from .utils import get_user_profile, update_user_profile


class UserProfileView(LoginRequiredMixin, DetailView):
    """Display user profile page with settings and favorites cards"""
    model = User
    template_name = 'users/user_detail_page.html'
    context_object_name = 'user'
    
    def get_object(self):
        # Return the current logged-in user
        return self.request.user
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Ensure user has a profile
        profile = get_user_profile(self.request.user)
        context['profile'] = profile
        return context


class UserSettingsView(LoginRequiredMixin, UpdateView):
    """User settings page for updating profile information"""
    model = UserProfile
    template_name = 'users/user_settings.html'
    fields = ['bio', 'location', 'date_of_birth', 'avatar', 'preferred_genres', 'reading_goals']
    success_url = reverse_lazy('users:profile')
    
    def get_object(self):
        # Get or create profile for current user
        return get_user_profile(self.request.user)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['user'] = self.request.user
        return context
    
    def form_valid(self, form):
        messages.success(self.request, 'Your profile has been updated successfully!')
        return super().form_valid(form)


class UserFavoritesView(LoginRequiredMixin, TemplateView):
    """Display user's favorite books"""
    template_name = 'users/user_favorites.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['user'] = self.request.user
        context['profile'] = get_user_profile(self.request.user)
        
        # Get user's favorite books (you'll need to implement this based on your books app)
        # For now, this is a placeholder
        context['favorite_books'] = []
        
        return context


class UserAccountView(LoginRequiredMixin, TemplateView):
    """General account management view"""
    template_name = 'users/user_account.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['user'] = self.request.user
        context['profile'] = get_user_profile(self.request.user)
        return context


@login_required
def update_profile_ajax(request):
    """AJAX view for updating profile fields"""
    if request.method == 'POST':
        field = request.POST.get('field')
        value = request.POST.get('value')
        
        if field and hasattr(UserProfile, field):
            try:
                update_user_profile(request.user, **{field: value})
                return JsonResponse({'success': True, 'message': 'Profile updated successfully'})
            except Exception as e:
                return JsonResponse({'success': False, 'message': str(e)})
    
    return JsonResponse({'success': False, 'message': 'Invalid request'})


# Function-based view alternative for user profile
@login_required
def user_profile_view(request):
    """Function-based view for user profile"""
    profile = get_user_profile(request.user)
    context = {
        'user': request.user,
        'profile': profile,
    }
    return render(request, 'users/user_detail_page.html', context)


@login_required
def user_settings_view(request):
    """Function-based view for user settings"""
    profile = get_user_profile(request.user)
    
    if request.method == 'POST':
        # Handle form submission
        bio = request.POST.get('bio', '')
        location = request.POST.get('location', '')
        preferred_genres = request.POST.get('preferred_genres', '')
        reading_goals = request.POST.get('reading_goals', 12)
        
        try:
            reading_goals = int(reading_goals)
        except (ValueError, TypeError):
            reading_goals = 12
        
        # Update profile
        profile.bio = bio
        profile.location = location
        profile.preferred_genres = preferred_genres
        profile.reading_goals = reading_goals
        
        if 'avatar' in request.FILES:
            profile.avatar = request.FILES['avatar']
        
        profile.save()
        messages.success(request, 'Your profile has been updated successfully!')
        return redirect('users:profile')
    
    context = {
        'user': request.user,
        'profile': profile,
    }
    return render(request, 'users/user_settings.html', context)


@login_required
def user_favorites_view(request):
    """Function-based view for user favorites"""
    profile = get_user_profile(request.user)
    
    # TODO: Implement favorites logic based on your books app
    favorite_books = []
    
    context = {
        'user': request.user,
        'profile': profile,
        'favorite_books': favorite_books,
    }
    return render(request, 'users/user_favorites.html', context)