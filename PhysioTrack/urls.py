from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('upload/', views.upload_image, name='upload'),
    path('result/<int:pk>/', views.result_view, name='result'),
    path('history/', views.history_view, name='history'),
    path('doctor/dashboard/', views.doctor_dashboard, name='doctor_dashboard'),
    path('doctor/approve/<int:pk>/', views.approve_doctor, name='approve_doctor'),
    path('doctor/reject/<int:pk>/', views.reject_doctor, name='reject_doctor'),
    path('pending-approval/', views.pending_approval, name='pending_approval'),
    path('profile/', views.profile_view, name='profile'),
    path('profile/change-password/', views.change_password, name='change_password'),
]
