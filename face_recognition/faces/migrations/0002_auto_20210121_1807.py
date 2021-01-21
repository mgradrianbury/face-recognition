# Generated by Django 3.1.5 on 2021-01-21 18:07

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('faces', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='faceimage',
            name='embedding',
            field=models.BinaryField(default=b''),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='facelabel',
            name='label',
            field=models.SlugField(max_length=200),
        ),
    ]
