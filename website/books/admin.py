from django.contrib import admin

from wagtail.snippets.models import register_snippet
from wagtail.snippets.views.snippets import SnippetViewSet

from .models import Book

@register_snippet
class BookViewSet(SnippetViewSet):
    model = Book
    menu_label = "Books"  # Label shown in the admin menu
    icon = "folder-inverse"  # Icon shown in the admin menu
    list_display = (
        'book_id', "title", "isbn13",
        "language_code", "publication_year", "average_rating",
        'artemis_emotion_embedding') # Fields shown in list view
    list_filter = ("publication_year", "language_code") # Fields available for filtering
    search_fields = ("title", "authors", "isbn", "isbn13") # Fields used for searching
    add_to_admin_menu = True  # Whether to add this snippet to the admin menu
    list_per_page = 20