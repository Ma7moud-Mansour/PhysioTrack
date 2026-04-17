import os
from functools import wraps

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.forms import AuthenticationForm, PasswordChangeForm
from django.conf import settings
from django.http import HttpResponseForbidden
from django.views.decorators.http import require_POST
from django.utils import timezone

from .models import PostureVideo, UserProfile
from .forms import RegisterForm, ImageUploadForm, UserUpdateForm, UserProfileUpdateForm
from .posture_analysis import analyze_posture


# ── Decorators ───────────────────────────────────────────────────────────────

def approved_required(view_func):
    """Block unapproved doctors from accessing protected views.
    Patients (auto-approved) pass through. Unapproved doctors are redirected."""
    @wraps(view_func)
    @login_required
    def wrapper(request, *args, **kwargs):
        profile = request.user.userprofile
        if profile.role == 'doctor' and not profile.is_approved():
            return redirect('pending_approval')
        return view_func(request, *args, **kwargs)
    return wrapper


def approved_doctor_required(view_func):
    """Only approved doctors can access this view."""
    @wraps(view_func)
    @login_required
    def wrapper(request, *args, **kwargs):
        profile = request.user.userprofile
        if profile.role != 'doctor' or not profile.is_approved():
            messages.error(request, 'Access denied. Only approved doctors can access this page.')
            return redirect('home')
        return view_func(request, *args, **kwargs)
    return wrapper


# ── Public Views ─────────────────────────────────────────────────────────────

def home(request):
    """Landing page."""
    return render(request, 'PhysioTrack/home.html')


def register_view(request):
    """User registration view with file upload support."""
    if request.user.is_authenticated:
        return redirect('home')

    if request.method == 'POST':
        form = RegisterForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            role = form.cleaned_data['role']
            if role == 'doctor':
                messages.info(
                    request,
                    'Your account has been created and is under review. '
                    'You will be able to log in once an approved doctor verifies your credentials.'
                )
            else:
                messages.success(request, 'Account created successfully! Please log in.')
            return redirect('login')
    else:
        form = RegisterForm()

    return render(request, 'PhysioTrack/register.html', {'form': form})


def login_view(request):
    """User login view — redirects unapproved doctors to pending page."""
    if request.user.is_authenticated:
        return redirect('home')

    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            profile = user.userprofile

            # Gate unapproved doctors
            if profile.role == 'doctor' and not profile.is_approved():
                return redirect('pending_approval')

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


# ── Pending Approval ─────────────────────────────────────────────────────────

@login_required
def pending_approval(request):
    """Show pending/rejected status page for unapproved doctors."""
    profile = request.user.userprofile
    # If already approved or is a patient, go home
    if profile.role == 'patient' or profile.is_approved():
        return redirect('home')

    return render(request, 'PhysioTrack/pending_approval.html', {
        'profile': profile,
    })


# ── Protected Patient/Doctor Views ───────────────────────────────────────────

@approved_required
def upload_image(request):
    """Handle image upload, run posture analysis, and save results."""
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            posture_image = form.save(commit=False)
            posture_image.user = request.user
            posture_image.save()

            image_path = os.path.join(settings.MEDIA_ROOT, str(posture_image.image))
            analysis = analyze_posture(image_path)

            posture_image.result = analysis['result']
            posture_image.score = analysis['score']
            posture_image.save()

            request.session['posture_issues'] = analysis.get('issues', [])
            request.session['visualization_image'] = analysis.get('visualization_image')
            request.session['posture_message'] = analysis.get('message')
            request.session['neck_score'] = analysis.get('neck_score')
            request.session['back_score'] = analysis.get('back_score')

            messages.success(request, 'Image uploaded and analyzed successfully!')
            return redirect('result', pk=posture_image.pk)
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, error)
    else:
        form = ImageUploadForm()

    return render(request, 'PhysioTrack/upload.html', {'form': form})


@approved_required
def result_view(request, pk):
    """Display analysis result for a specific video."""
    posture_video = get_object_or_404(PostureVideo, pk=pk, user=request.user)
    issues = request.session.pop('posture_issues', [])
    visualization_image = request.session.pop('visualization_image', None)
    message = request.session.pop('posture_message', None)
    neck_score = request.session.pop('neck_score', None)
    back_score = request.session.pop('back_score', None)

    return render(request, 'PhysioTrack/result.html', {
        'video': posture_video,
        'issues': issues,
        'visualization_image': visualization_image,
        'message': message,
        'neck_score': neck_score,
        'back_score': back_score,
    })


@approved_required
def history_view(request):
    """Display all past analyses for the logged-in user, newest first."""
    videos = PostureVideo.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'PhysioTrack/history.html', {'videos': videos})


# ── Doctor Dashboard & Approval ──────────────────────────────────────────────

@approved_doctor_required
def doctor_dashboard(request):
    """Dashboard for approved doctors: patients list + pending doctor applications."""
    profile = request.user.userprofile

    # This doctor's patients
    patients = profile.patients.all()
    for patient in patients:
        patient.latest_result = PostureVideo.objects.filter(
            user=patient.user
        ).order_by('-created_at').first()

    # Pending doctor applications
    pending_doctors = UserProfile.objects.filter(
        role='doctor', status='pending'
    ).select_related('user').order_by('-created_at')

    context = {
        'patients': patients,
        'pending_doctors': pending_doctors,
    }

    return render(request, 'PhysioTrack/doctor_dashboard.html', context)


@approved_doctor_required
@require_POST
def approve_doctor(request, pk):
    """Approve a pending doctor application. POST only."""
    doctor_profile = get_object_or_404(UserProfile, pk=pk, role='doctor', status='pending')
    doctor_profile.status = 'approved'
    doctor_profile.approved_by = request.user
    doctor_profile.approved_at = timezone.now()
    doctor_profile.save()

    messages.success(request, f'Dr. {doctor_profile.name} has been approved.')
    return redirect('doctor_dashboard')


@approved_doctor_required
@require_POST
def reject_doctor(request, pk):
    """Reject a pending doctor application. POST only. Does NOT delete the user."""
    doctor_profile = get_object_or_404(UserProfile, pk=pk, role='doctor', status='pending')
    doctor_profile.status = 'rejected'
    doctor_profile.save()

    messages.warning(request, f'{doctor_profile.name} has been rejected.')
    return redirect('doctor_dashboard')


# ── Profile & Password ───────────────────────────────────────────────────────

@login_required
def profile_view(request):
    """User profile view."""
    user = request.user
    profile = user.userprofile
    is_doctor = profile.role == 'doctor'

    if request.method == 'POST':
        user_form = UserUpdateForm(request.POST, instance=user)
        profile_form = UserProfileUpdateForm(request.POST, instance=profile)

        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
            messages.success(request, 'Profile updated successfully!')
            return redirect('profile')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        user_form = UserUpdateForm(instance=user)
        profile_form = UserProfileUpdateForm(instance=profile)

    patients = None
    if is_doctor:
        patients = profile.patients.all()

    context = {
        'user_form': user_form,
        'profile_form': profile_form,
        'is_doctor': is_doctor,
        'patients': patients,
    }
    return render(request, 'PhysioTrack/profile.html', context)


@login_required
def change_password(request):
    """Handle password changes for authenticated users."""
    if request.method == 'POST':
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)
            messages.success(request, 'Your password was successfully updated!')
            return redirect('profile')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = PasswordChangeForm(request.user)

    return render(request, 'PhysioTrack/change_password.html', {'form': form})