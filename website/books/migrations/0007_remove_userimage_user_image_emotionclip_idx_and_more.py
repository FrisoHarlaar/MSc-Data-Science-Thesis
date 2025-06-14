# Generated by Django 5.1.9 on 2025-06-09 17:51

import pgvector.django.vector
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('books', '0006_userimage_book_emotionclip_idx_and_more'),
    ]

    operations = [
        migrations.RemoveIndex(
            model_name='userimage',
            name='user_image_emotionclip_idx',
        ),
        migrations.RemoveField(
            model_name='userimage',
            name='emotionclip_embedding',
        ),
        migrations.AddField(
            model_name='book',
            name='artemis_confidence',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='book',
            name='artemis_emotion_embedding',
            field=pgvector.django.vector.VectorField(blank=True, dimensions=9, null=True),
        ),
        migrations.AddField(
            model_name='book',
            name='artemis_predicted_emotion',
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
        migrations.AddField(
            model_name='book',
            name='bert_confidence',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='book',
            name='bert_emotion_embedding',
            field=pgvector.django.vector.VectorField(blank=True, dimensions=9, null=True),
        ),
        migrations.AddField(
            model_name='book',
            name='bert_predicted_emotion',
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
        migrations.AddField(
            model_name='book',
            name='multimodal_emotion_embedding',
            field=pgvector.django.vector.VectorField(blank=True, dimensions=9, null=True),
        ),
        migrations.AddField(
            model_name='userimage',
            name='artemis_emotion_embedding',
            field=pgvector.django.vector.VectorField(blank=True, dimensions=9, null=True),
        ),
        migrations.AddField(
            model_name='userimage',
            name='confidence',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='userimage',
            name='dominant_emotion',
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
        migrations.AddIndex(
            model_name='book',
            index=models.Index(fields=['artemis_emotion_embedding'], name='artemis_emotion_idx'),
        ),
        migrations.AddIndex(
            model_name='book',
            index=models.Index(fields=['bert_emotion_embedding'], name='bert_emotion_idx'),
        ),
        migrations.AddIndex(
            model_name='book',
            index=models.Index(fields=['multimodal_emotion_embedding'], name='multimodal_emotion_idx'),
        ),
        migrations.AddIndex(
            model_name='userimage',
            index=models.Index(fields=['artemis_emotion_embedding'], name='user_image_emotion_idx'),
        ),
    ]
