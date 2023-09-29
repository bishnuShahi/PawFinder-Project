from django.db import models
from django.contrib.auth.models import User

class MyModel(models.Model):
    image = models.ImageField(upload_to='image', null=True, blank=True)
    class Meta:
        app_label = 'Authenticator'
