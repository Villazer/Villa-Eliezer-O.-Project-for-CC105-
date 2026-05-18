from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True
    dependencies = []
    operations = [
        migrations.CreateModel(
            name='ClassificationLog',
            fields=[
                ('id',               models.BigAutoField(auto_created=True, primary_key=True, serialize=False)),
                ('game_title',       models.CharField(blank=True, default='', max_length=300)),
                ('rating',           models.CharField(max_length=4)),
                ('confidence',       models.FloatField()),
                ('all_probs_json',   models.TextField()),
                ('model_used',       models.CharField(max_length=300)),
                ('descriptors_json', models.TextField(default='[]')),
                ('genre',            models.CharField(blank=True, default='', max_length=100)),
                ('platform',         models.CharField(blank=True, default='', max_length=100)),
                ('critic_score',     models.FloatField(blank=True, null=True)),
                ('created_at',       models.DateTimeField(auto_now_add=True)),
            ],
            options={'ordering': ['-created_at']},
        ),
    ]
