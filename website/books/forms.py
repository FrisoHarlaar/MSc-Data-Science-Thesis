from django import forms

class ImageUploadForm(forms.Form):
    image = forms.ImageField(
        label='Upload an image',
        help_text='Upload an image to find books with similar emotional content'
    )
    
    limit = forms.IntegerField(
        label='Number of results',
        min_value=1,
        max_value=20,
        initial=6,
        required=False
    )
    
    APPROACH_CHOICES = [
        ('multimodal', 'Multimodal (Best)'),
        ('artemis', 'Visual Emotions Only'),
        ('bert', 'Text Emotions Only'),
    ]
    
    approach = forms.ChoiceField(
        label='Matching approach',
        choices=APPROACH_CHOICES,
        required=False,
        initial='multimodal',
        widget=forms.RadioSelect
    )