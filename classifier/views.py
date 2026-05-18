import json, time
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.core.paginator import Paginator

from . import model_loader
from .models import ClassificationLog

FEATURES = [
    'alcohol_reference','animated_blood','blood','blood_and_gore',
    'cartoon_violence','crude_humor','drug_reference','fantasy_violence',
    'intense_violence','language','lyrics','mature_humor','mild_blood',
    'mild_cartoon_violence','mild_fantasy_violence','mild_language',
    'mild_lyrics','mild_suggestive_themes','mild_violence','no_descriptors',
    'nudity','partial_nudity','sexual_content','sexual_themes',
    'simulated_gambling','strong_janguage','strong_sexual_content',
    'suggestive_themes','use_of_alcohol','use_of_drugs_and_alcohol','violence',
]

FEATURE_LABELS = {
    'alcohol_reference':'Alcohol Reference','animated_blood':'Animated Blood',
    'blood':'Blood','blood_and_gore':'Blood and Gore','cartoon_violence':'Cartoon Violence',
    'crude_humor':'Crude Humor','drug_reference':'Drug Reference',
    'fantasy_violence':'Fantasy Violence','intense_violence':'Intense Violence',
    'language':'Language','lyrics':'Lyrics','mature_humor':'Mature Humor',
    'mild_blood':'Mild Blood','mild_cartoon_violence':'Mild Cartoon Violence',
    'mild_fantasy_violence':'Mild Fantasy Violence','mild_language':'Mild Language',
    'mild_lyrics':'Mild Lyrics','mild_suggestive_themes':'Mild Suggestive Themes',
    'mild_violence':'Mild Violence','no_descriptors':'No Descriptors',
    'nudity':'Nudity','partial_nudity':'Partial Nudity',
    'sexual_content':'Sexual Content','sexual_themes':'Sexual Themes',
    'simulated_gambling':'Simulated Gambling','strong_janguage':'Strong Language',
    'strong_sexual_content':'Strong Sexual Content','suggestive_themes':'Suggestive Themes',
    'use_of_alcohol':'Use of Alcohol','use_of_drugs_and_alcohol':'Use of Drugs and Alcohol',
    'violence':'Violence',
}

RATING_INFO = {
    'E':  {'label':'Everyone',       'color':'#10b981','bg':'#ecfdf5','border':'#059669','desc':'Content suitable for all ages.'},
    'ET': {'label':'Everyone 10+',   'color':'#0ea5e9','bg':'#f0f9ff','border':'#0284c7','desc':'May contain mild content not suitable for young children.'},
    'T':  {'label':'Teen',           'color':'#f97316','bg':'#fff7ed','border':'#ea580c','desc':'Content suitable for ages 13 and up.'},
    'M':  {'label':'Mature 17+',     'color':'#f43f5e','bg':'#fff1f2','border':'#e11d48','desc':'Content suitable for ages 17 and up.'},
    'AO': {'label':'Adults Only 18+','color':'#a855f7','bg':'#faf5ff','border':'#9333ea','desc':'Content suitable only for adults.'},
}

GENRES = ['Action','Adventure','Fighting','Misc','Platform','Puzzle',
          'Racing','Role-Playing','Shooter','Simulation','Sports','Strategy']

PLATFORMS = ['PC','PS5','PS4','PS3','PS2','Xbox Series X','Xbox One',
             'Xbox 360','Nintendo Switch','Wii','Wii U','Nintendo DS',
             '3DS','Mobile','Other']


def index(request):
    active      = model_loader.get_active_model_name()
    model_info  = model_loader.get_model_info()
    total       = ClassificationLog.objects.count()
    return render(request, 'classifier/index.html', {
        'features':        FEATURES,
        'feature_labels':  FEATURE_LABELS,
        'feature_groups':  _group_features(),
        'active_model':    active,
        'model_info':      model_info,
        'rating_info':     RATING_INFO,
        'genres':          GENRES,
        'platforms':       PLATFORMS,
        'total_classified':total,
    })


@csrf_exempt
@require_POST
def classify(request):
    try:
        body   = json.loads(request.body)
        title  = body.get('game_title', '').strip()[:300]
        values = [int(body.get(f, 0)) for f in FEATURES]

        extra = {
            'genre':           body.get('genre', ''),
            'platform':        body.get('platform', ''),
            'critic_score':    body.get('critic_score'),
            'year_of_release': body.get('year_of_release'),
            'publisher':       body.get('publisher', ''),
        }

        t0 = time.perf_counter()
        result = model_loader.predict(values, extra=extra)
        runtime_ms = round((time.perf_counter() - t0) * 1000)

        result['rating_info'] = RATING_INFO.get(result['rating'], {})
        result['runtime_ms']  = runtime_ms

        active_descriptors = [FEATURE_LABELS[f] for f, v in zip(FEATURES, values) if v == 1]

        log = ClassificationLog.objects.create(
            game_title       = title,
            rating           = result['rating'],
            confidence       = result['confidence'],
            all_probs_json   = json.dumps(result['all_probs']),
            model_used       = result['model_used'],
            descriptors_json = json.dumps(active_descriptors),
            genre            = extra.get('genre','') or '',
            platform         = extra.get('platform','') or '',
            critic_score     = float(extra['critic_score']) if extra.get('critic_score') not in (None,'') else None,
        )
        result['log_id'] = log.pk
        return JsonResponse({'success': True, 'result': result})

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


def logs(request):
    qs = ClassificationLog.objects.all()

    rating_filter = request.GET.get('rating', '')
    if rating_filter:
        qs = qs.filter(rating=rating_filter)
    search = request.GET.get('q', '').strip()
    if search:
        qs = qs.filter(game_title__icontains=search)

    paginator = Paginator(qs, 20)
    page_obj  = paginator.get_page(request.GET.get('page'))

    all_logs  = ClassificationLog.objects.all()
    total     = all_logs.count()
    by_rating = {r: all_logs.filter(rating=r).count() for r in ['E','ET','T','M']}

    avg_conf = None
    if total:
        from django.db.models import Avg
        avg_conf = round(all_logs.aggregate(Avg('confidence'))['confidence__avg'] or 0, 1)

    return render(request, 'classifier/logs.html', {
        'page_obj':      page_obj,
        'rating_filter': rating_filter,
        'search':        search,
        'total':         total,
        'by_rating':     by_rating,
        'avg_conf':      avg_conf,
        'rating_info':   RATING_INFO,
    })


def log_detail(request, pk):
    log = get_object_or_404(ClassificationLog, pk=pk)
    return JsonResponse({
        'id':          log.pk,
        'game_title':  log.game_title or '(untitled)',
        'rating':      log.rating,
        'confidence':  log.confidence,
        'all_probs':   log.all_probs,
        'model_used':  log.model_used,
        'descriptors': log.descriptors,
        'genre':       log.genre,
        'platform':    log.platform,
        'critic_score':log.critic_score,
        'created_at':  log.created_at.strftime('%Y-%m-%d %H:%M:%S'),
        'rating_info': RATING_INFO.get(log.rating, {}),
    })


@csrf_exempt
@require_POST
def log_delete(request, pk):
    get_object_or_404(ClassificationLog, pk=pk).delete()
    return JsonResponse({'success': True})


def manage_model(request):
    models  = model_loader.list_models()
    active  = model_loader.get_active_model_name()
    message = None
    if request.method == 'POST':
        selected = request.POST.get('model')
        if selected and selected in models:
            model_loader.set_active_model(selected)
            active  = selected
            message = ('success', f'Active model switched to: {selected}')
        else:
            message = ('error', 'Invalid model selection.')

    model_infos = {}
    for m in models:
        try:
            model_infos[m] = model_loader.get_model_info(m)
        except Exception:
            model_infos[m] = {}

    return render(request, 'classifier/manage_model.html', {
        'models':      models,
        'model_infos': model_infos,
        'active_model':active,
        'message':     message,
    })


def _group_features():
    groups = {'Violence':[], 'Language':[], 'Sexual Content':[], 'Substances':[], 'Other':[]}
    for f in FEATURES:
        lbl = FEATURE_LABELS[f]
        if any(x in f for x in ['violence','blood','gore','intense']):
            groups['Violence'].append((f, lbl))
        elif any(x in f for x in ['language','lyrics','humor']):
            groups['Language'].append((f, lbl))
        elif any(x in f for x in ['sexual','nudity','suggestive']):
            groups['Sexual Content'].append((f, lbl))
        elif any(x in f for x in ['alcohol','drug']):
            groups['Substances'].append((f, lbl))
        else:
            groups['Other'].append((f, lbl))
    return groups
