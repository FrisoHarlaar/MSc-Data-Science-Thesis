from django.contrib.auth.models import AbstractUser
from django.db import models
from wagtail.users.models import UserProfile


class User(AbstractUser):
    """
    Custom user model that extends Django's AbstractUser.
    Add any additional fields you need here.
    """
    # You can add custom fields here, for example:
    # date_of_birth = models.DateField(null=True, blank=True)
    # bio = models.TextField(max_length=500, blank=True)
    # location = models.CharField(max_length=30, blank=True)
    
    pass