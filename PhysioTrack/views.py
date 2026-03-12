import os
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.contrib import messages
from django.conf import settings

from .models import PostureVideo
from .forms import RegisterForm, ImageUploadForm
from .posture_analysis import analyze_posture


def home(request):
    """Landing page."""
    return render(request, 'PhysioTrack/home.html')


def register_view(request):
    """User registration view."""
    if request.user.is_authenticated:
        return redirect('home')

    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Account created successfully! Please log in.')
            return redirect('login')
    else:
        form = RegisterForm()

    return render(request, 'PhysioTrack/register.html', {'form': form})


def login_view(request):
    """User login view."""
    if request.user.is_authenticated:
        return redirect('home')

    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, f'Welcome back, {user.username}!')
            return redirect('home')
    else:
        form = AuthenticationForm()

    return render(request, 'PhysioTrack/login.html', {'form': form})


def logout_view(request):
    """Log out the user and redirect to home."""
    logout(request)
    messages.info(request, 'You have been logged out.')
    return redirect('home')


@login_required
def upload_image(request):
    """Handle image upload, run posture analysis, and save results."""
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the PostureVideo instance (without result yet)
            posture_image = form.save(commit=False)
            posture_image.user = request.user
            posture_image.save()

            # Get the absolute path to the saved image file
            image_path = os.path.join(settings.MEDIA_ROOT, str(posture_image.image))

            # Run posture analysis
            analysis = analyze_posture(image_path)

            # Store results
            posture_image.result = analysis['result']
            posture_image.score = analysis['score']
            posture_image.save()

            # Store detected posture issues in session for the result page
            request.session['posture_issues'] = analysis.get('issues', [])
            request.session['visualization_image'] = analysis.get('visualization_image')
            request.session['posture_message'] = analysis.get('message')

            messages.success(request, 'Image uploaded and analyzed successfully!')
            return redirect('result', pk=posture_image.pk)
        else:
            # Form validation errors
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, error)
    else:
        form = ImageUploadForm()

    return render(request, 'PhysioTrack/upload.html', {'form': form})


@login_required
def result_view(request, pk):
    """Display analysis result for a specific video."""
    posture_video = get_object_or_404(PostureVideo, pk=pk, user=request.user)
    issues = request.session.pop('posture_issues', [])
    visualization_image = request.session.pop('visualization_image', None)
    message = request.session.pop('posture_message', None)
    
    return render(request, 'PhysioTrack/result.html', {
        'video': posture_video,
        'issues': issues,
        'visualization_image': visualization_image,
        'message': message,
    })


@login_required
def history_view(request):
    """Display all past analyses for the logged-in user, newest first."""
    videos = PostureVideo.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'PhysioTrack/history.html', {'videos': videos})