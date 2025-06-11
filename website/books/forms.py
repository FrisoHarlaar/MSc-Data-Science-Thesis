from django import forms

class ImageUploadForm(forms.Form):
    image = forms.ImageField(
        widget=forms.FileInput(attrs={
            'style': 'display: none;',
            'accept': 'image/*'
        })
    )