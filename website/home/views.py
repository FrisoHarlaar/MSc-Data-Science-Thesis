from django.views.generic import TemplateView


class LandingPageView(TemplateView):
    """Landing page view that renders the home template"""
    template_name = 'home.html'