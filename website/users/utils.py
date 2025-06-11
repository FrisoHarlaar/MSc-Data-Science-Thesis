from django.contrib.auth.models import User
from .models import UserProfile


def get_user_profile(user):
    """Get or create user profile"""
    profile, created = UserProfile.objects.get_or_create(user=user)
    return profile


def update_user_profile(user, **kwargs):
    """Update user profile with given kwargs"""
    profile = get_user_profile(user)
    for key, value in kwargs.items():
        if hasattr(profile, key):
            setattr(profile, key, value)
    profile.save()
    return profile