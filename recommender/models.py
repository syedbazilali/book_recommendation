from django.db import models

class Book(models.Model):
    title = models.CharField(max_length=255)
    author = models.CharField(max_length=255)
    desc = models.TextField()
    genre = models.TextField()
    img = models.URLField()
    link = models.URLField()
    rating = models.FloatField()
    totalratings = models.IntegerField()

    def __str__(self):
        return self.title
