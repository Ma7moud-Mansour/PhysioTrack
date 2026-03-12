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


class VideoUploadForm(forms.ModelForm):
    """Form for uploading posture analysis videos with validation."""

    ALLOWED_EXTENSIONS = ['mp4', 'mov', 'avi']
    MAX_DURATION_SECONDS = 20

    class Meta:
        model = PostureVideo
        fields = ('video',)
        widgets = {
            'video': forms.ClearableFileInput(attrs={
                'class': 'form-input file-input',
                'accept': '.mp4,.mov,.avi',
            })
        }

    def clean_video(self):
        video = self.cleaned_data.get('video')
        if not video:
            raise forms.ValidationError('Please select a video file.')

        # Validate file extension
        ext = os.path.splitext(video.name)[1].lower().lstrip('.')
        if ext not in self.ALLOWED_EXTENSIONS:
            raise forms.ValidationError(
                f'Invalid format "{ext}". Accepted formats: {", ".join(self.ALLOWED_EXTENSIONS)}.'
            )

        # Validate video duration using OpenCV
        # Write to a temporary file so OpenCV can read it
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=f'.{ext}', delete=False) as tmp:
            for chunk in video.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        try:
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                raise forms.ValidationError('Could not read the video file. Please upload a valid video.')

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()

            if fps > 0:
                duration = frame_count / fps
                if duration > self.MAX_DURATION_SECONDS:
                    raise forms.ValidationError(
                        f'Video is {duration:.1f}s long. Maximum allowed duration is {self.MAX_DURATION_SECONDS} seconds.'
                    )
            else:
                raise forms.ValidationError('Could not determine video duration. Please upload a valid video.')
        finally:
            os.unlink(tmp_path)

        # Reset file pointer so Django can save it
        video.seek(0)
        return video
