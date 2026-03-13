import os
import cv2
from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import PostureVideo, UserProfile


class RegisterForm(UserCreationForm):
    """Registration form extending UserCreationForm with an email field."""
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'form-input',
            'placeholder': 'Email address',
        })
    )

    name = forms.CharField(
        max_length=100, required=True,
        widget=forms.TextInput(attrs={'class': 'form-input', 'placeholder': 'Full Name'})
    )
    phone_number = forms.CharField(
        max_length=20, required=False,
        widget=forms.TextInput(attrs={'class': 'form-input', 'placeholder': 'Phone Number (Optional)'})
    )
    age = forms.IntegerField(
        required=False,
        widget=forms.NumberInput(attrs={'class': 'form-input', 'placeholder': 'Age (Optional)', 'id': 'id_age'})
    )
    activity = forms.ChoiceField(
        required=False,
        choices=[
            ("", "Select Primary Activity"),
            ("laptop", "Working on laptop"),
            ("desktop", "Desktop work"),
            ("phone", "Using phone"),
            ("studying", "Studying"),
            ("gaming", "Gaming")
        ],
        widget=forms.Select(attrs={'class': 'form-input', 'id': 'id_activity'})
    )
    sitting_hours = forms.ChoiceField(
        required=False,
        choices=[
            ("", "Select Sitting Hours"),
            ("1-3", "1–3 hours"),
            ("4-6", "4–6 hours"),
            ("7-9", "7–9 hours"),
            ("10+", "10+ hours")
        ],
        widget=forms.Select(attrs={'class': 'form-input', 'id': 'id_sitting_hours'})
    )
    exercise_habit = forms.ChoiceField(
        required=False,
        choices=[
            ("", "Select Exercise Habit"),
            ("regular", "Yes regularly"),
            ("sometimes", "Sometimes"),
            ("rarely", "Rarely"),
            ("never", "Never")
        ],
        widget=forms.Select(attrs={'class': 'form-input', 'id': 'id_exercise_habit'})
    )

    role = forms.ChoiceField(
        required=True,
        choices=[
            ('patient', 'Patient'),
            ('doctor', 'Doctor')
        ],
        widget=forms.Select(attrs={'class': 'form-input', 'id': 'id_role'})
    )
    
    doctor = forms.ModelChoiceField(
        queryset=UserProfile.objects.filter(role='doctor'),
        required=False,
        empty_label="Select your Doctor (Optional)",
        widget=forms.Select(attrs={'class': 'form-input', 'id': 'doctor-field'})
    )

    class Meta:
        model = User
        fields = ('username', 'email', 'name', 'phone_number', 'role', 'doctor', 'password1', 'password2', 'age', 'activity', 'sitting_hours', 'exercise_habit')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Premium Tailwind CSS classes for all inputs
        input_classes = 'w-full px-4 py-3.5 rounded-xl border-2 border-slate-200/80 bg-slate-50 text-slate-900 text-sm font-semibold focus:ring-4 focus:ring-primary/10 focus:border-primary focus:bg-white outline-none transition-all duration-300 hover:border-slate-300 shadow-sm appearance-none'
        
        for field_name, field in self.fields.items():
            field.widget.attrs['class'] = input_classes
            
        self.fields['username'].widget.attrs['placeholder'] = 'Choose a username'
        self.fields['password1'].widget.attrs['placeholder'] = 'Create a secure password'
        self.fields['password2'].widget.attrs['placeholder'] = 'Confirm your password'

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
            profile = UserProfile.objects.create(
                user=user,
                name=self.cleaned_data['name'],
                phone_number=self.cleaned_data.get('phone_number', ''),
                age=self.cleaned_data['age'],
                activity=self.cleaned_data['activity'],
                sitting_hours=self.cleaned_data['sitting_hours'],
                exercise_habit=self.cleaned_data['exercise_habit'],
                role=self.cleaned_data['role']
            )
            if self.cleaned_data.get('doctor'):
                profile.doctor = self.cleaned_data['doctor']
                profile.save()
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


class UserUpdateForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['email']
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        input_classes = 'w-full px-4 py-3.5 rounded-xl border-2 border-slate-200/80 bg-slate-50 text-slate-900 text-sm font-semibold focus:ring-4 focus:ring-primary/10 focus:border-primary focus:bg-white outline-none transition-all duration-300 hover:border-slate-300 shadow-sm appearance-none'
        for field_name, field in self.fields.items():
            field.widget.attrs['class'] = input_classes

class UserProfileUpdateForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ['name', 'phone_number', 'age', 'activity', 'sitting_hours', 'exercise_habit', 'doctor']
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        input_classes = 'w-full px-4 py-3.5 rounded-xl border-2 border-slate-200/80 bg-slate-50 text-slate-900 text-sm font-semibold focus:ring-4 focus:ring-primary/10 focus:border-primary focus:bg-white outline-none transition-all duration-300 hover:border-slate-300 shadow-sm appearance-none'
        for field_name, field in self.fields.items():
            field.widget.attrs['class'] = input_classes
            
        # specifically for the UI ID binding to hide them in JS
        if 'age' in self.fields: self.fields['age'].widget.attrs['id'] = 'id_age'
        if 'activity' in self.fields: self.fields['activity'].widget.attrs['id'] = 'id_activity'
        if 'sitting_hours' in self.fields: self.fields['sitting_hours'].widget.attrs['id'] = 'id_sitting_hours'
        if 'exercise_habit' in self.fields: self.fields['exercise_habit'].widget.attrs['id'] = 'id_exercise_habit'
