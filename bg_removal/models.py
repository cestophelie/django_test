from django.db import models


# Create your models here.
class Post(models.Model):
    title = models.CharField(default='title', max_length=200)
    category = models.CharField(default='category', max_length=200)
    image = models.ImageField(default='media/default_image.jpg')


class Accounts(models.Model):
    identify = models.CharField(default='identify', max_length=20)
    password = models.CharField(default='password', max_length=20)

# class bg_removal(models.Model):  # 제목이 camel case convention
#     Title = models.CharField('TITLE', max_length=50)
#     Content = models.TextField('CONTENT')
#
#     def __str__(self):
#         return self.Title
