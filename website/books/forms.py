from django import forms

class ImageUploadForm(forms.Form):
    image = forms.ImageField(
        label='Upload an image',
        help_text='Upload an image to find books with similar visual style and emotions'
    )
    
    limit = forms.IntegerField(
        label='Number of results',
        min_value=1,
        max_value=20,
        initial=6,
        required=False
    )
    
    EMOTION_CHOICES = [
        ('all', 'All emotions'),
        ('amusement', 'Amusement'), 
        ('awe', 'Awe'),
        ('contentment', 'Contentment'),
        ('excitement', 'Excitement'),
        ('anger', 'Anger'), 
        ('disgust', 'Disgust'),
        ('fear', 'Fear'),
        ('sadness', 'Sadness')
    ]
    
    emotion_filter = forms.ChoiceField(
        label='Filter by emotion',
        choices=EMOTION_CHOICES,
        required=False,
        initial='all'
    )