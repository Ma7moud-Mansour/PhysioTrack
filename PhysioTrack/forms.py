import os
import cv2
from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import PostureVideo


class RegisterForm(UserCreationForm):
    """Registration form extending UserCreationForm with an email field."""
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'form-input',
            'placeholder': 'Email address',
        })
    )

    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['username'].widget.attrs.update({
            'class': 'form-input',
            'placeholder': 'Username',
        })
        self.fields['password1'].widget.attrs.update({
            'class': 'form-input',
            'placeholder': 'Password',
        })
        self.fields['password2'].widget.attrs.update({
            'class': 'form-input',
            'placeholder': 'Confirm password',
        })

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
        return user


class ImageUploadForm(forms.ModelForm):
    """Form for uploading posture analysis images with validation."""

    ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'webp']

    class Meta:
        model = PostureVideo
        fields = ('image',)
        widgets = {
            'image': forms.ClearableFileInput(attrs={
                'class': 'form-input file-input',
                'accept': '.jpg,.jpeg,.png,.webp',
            })
        }

    def clean_image(self):
        image = self.cleaned_data.get('image')
        if not image:
            raise forms.ValidationError('Please select an image file.')

        # Validate file extension
        ext = os.path.splitext(image.name)[1].lower().lstrip('.')
        if ext not in self.ALLOWED_EXTENSIONS:
            raise forms.ValidationError(
                f'Invalid format "{ext}". Accepted formats: {", ".join(self.ALLOWED_EXTENSIONS)}.'
            )

        # Basic OpenCV validation to ensure the image is readable
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=f'.{ext}', delete=False) as tmp:
            for chunk in image.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        try:
            img = cv2.imread(tmp_path)
            if img is None:
                raise forms.ValidationError('Could not read the image file. Please upload a valid image.')
        finally:
            os.unlink(tmp_path)

        # Reset file pointer so Django can save it
        image.seek(0)
        return image
