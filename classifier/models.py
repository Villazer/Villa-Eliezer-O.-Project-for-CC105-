from django.db import models
import json


class ClassificationLog(models.Model):
    game_title       = models.CharField(max_length=300, blank=True, default='')
    rating           = models.CharField(max_length=4)
    confidence       = models.FloatField()
    all_probs_json   = models.TextField()
    model_used       = models.CharField(max_length=300)
    descriptors_json = models.TextField(default='[]')
    genre            = models.CharField(max_length=100, blank=True, default='')
    platform         = models.CharField(max_length=100, blank=True, default='')
    critic_score     = models.FloatField(null=True, blank=True)
    created_at       = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    @property
    def all_probs(self):
        try:   return json.loads(self.all_probs_json)
        except: return {}

    @property
    def descriptors(self):
        try:   return json.loads(self.descriptors_json)
        except: return []

    def __str__(self):
        return f'{self.game_title or "(untitled)"} → {self.rating} ({self.confidence}%)'
