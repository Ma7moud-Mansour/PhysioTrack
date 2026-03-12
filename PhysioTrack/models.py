from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    
    role = models.CharField(
        max_length=10,
        choices=[
            ('patient', 'Patient'),
            ('doctor', 'Doctor')
        ],
        default='patient'
    )
    
    doctor = models.ForeignKey(
        'self',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='patients',
        limit_choices_to={'role': 'doctor'}
    )
    
    name = models.CharField(max_length=100)
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

    def __str__(self):
        return f"{self.user.username}'s Profile"


class PostureVideo(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.FileField(upload_to='images/')
    result = models.CharField(max_length=50, blank=True)
    score = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)