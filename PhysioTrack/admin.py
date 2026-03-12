from django.contrib import admin
from .models import PostureVideo


@admin.register(PostureVideo)
class PostureVideoAdmin(admin.ModelAdmin):
    list_display = ('user', 'result', 'score', 'created_at')
    list_filter = ('result', 'created_at')
    search_fields = ('user__username',)
    ordering = ('-created_at',)
