from django.db import models
from django.contrib.auth.models import User


class UserProfile(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('approved', 'Approved'),
        ('rejected', 'Rejected'),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE)

    role = models.CharField(
        max_length=10,
        choices=[
            ('patient', 'Patient'),
            ('doctor', 'Doctor')
        ],
        default='patient'
    )

    status = models.CharField(
        max_length=10,
        choices=STATUS_CHOICES,
        default='approved',
        help_text='Patients are auto-approved. Doctors start as pending.',
    )

    verification_document = models.ImageField(
        upload_to='verification_docs/',
        blank=True,
        null=True,
        help_text='Required for doctors. Upload ID card or certificate.',
    )

    approved_by = models.ForeignKey(
        User,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='approved_doctors',
    )

    approved_at = models.DateTimeField(null=True, blank=True)

    doctor = models.ForeignKey(
        'self',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='patients',
        limit_choices_to={'role': 'doctor'}
    )

    name = models.CharField(max_length=100)
    phone_number = models.CharField(max_length=20, blank=True, null=True)
    age = models.IntegerField(null=True, blank=True)
    activity = models.CharField(max_length=50, choices=[
        ("laptop", "Working on laptop"),
        ("desktop", "Desktop work"),
        ("phone", "Using phone"),
        ("studying", "Studying"),
        ("gaming", "Gaming")
    ], blank=True)
    sitting_hours = models.CharField(max_length=20, choices=[
        ("1-3", "1–3 hours"),
        ("4-6", "4–6 hours"),
        ("7-9", "7–9 hours"),
        ("10+", "10+ hours")
    ], blank=True)
    exercise_habit = models.CharField(max_length=20, choices=[
        ("regular", "Yes regularly"),
        ("sometimes", "Sometimes"),
        ("rarely", "Rarely"),
        ("never", "Never")
    ], blank=True)

    created_at = models.DateTimeField(auto_now_add=True, null=True)

    # ── Helper methods ──

    def is_approved(self):
        """Single source of truth for approval status."""
        return self.status == 'approved'

    def is_pending(self):
        return self.status == 'pending'

    def is_rejected(self):
        return self.status == 'rejected'

    def __str__(self):
        return f"{self.user.username}'s Profile"


class PostureVideo(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.FileField(upload_to='images/')
    result = models.CharField(max_length=50, blank=True)
    score = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)