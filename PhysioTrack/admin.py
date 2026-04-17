from django.contrib import admin
from .models import PostureVideo, UserProfile


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'name', 'role', 'status', 'created_at')
    list_filter = ('role', 'status')
    search_fields = ('user__username', 'user__email', 'name')
    readonly_fields = ('approved_by', 'approved_at', 'created_at')
    list_editable = ('status',)


@admin.register(PostureVideo)
class PostureVideoAdmin(admin.ModelAdmin):
    list_display = ('user', 'result', 'score', 'created_at')
    list_filter = ('result', 'created_at')
    search_fields = ('user__username',)
    ordering = ('-created_at',)
