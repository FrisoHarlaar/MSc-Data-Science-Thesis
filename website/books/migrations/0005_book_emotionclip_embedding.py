# Generated by Django 5.1.9 on 2025-05-13 18:57

import pgvector.django.vector
from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('books', '0004_pgvector_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='book',
            name='emotionclip_embedding',
            field=pgvector.django.vector.VectorField(blank=True, dimensions=512, null=True),
        ),
    ]
