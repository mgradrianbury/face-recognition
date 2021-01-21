from django.contrib import admin
from django.utils.html import format_html

from .models import FaceImage, FaceLabel


class FaceImagesInline(admin.TabularInline):
    model = FaceImage
    extra = 0
    fields = ['original_image', 'preview_original_image', 'preview_face_image']
    readonly_fields = ['preview_original_image', 'preview_face_image']

    @staticmethod
    def preview_original_image(obj: FaceImage):
        return format_html('<img src="{}" width="150" />'.format(obj.original_image.url))

    @staticmethod
    def preview_face_image(obj: FaceImage):
        return format_html('<img src="{}" width="150" />'.format(obj.face_image.url))


@admin.register(FaceLabel)
class FaceLabelAdmin(admin.ModelAdmin):
    inlines = [FaceImagesInline]
    actions = ['create_embeddings']
