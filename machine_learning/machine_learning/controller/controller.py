from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from urllib.parse import urlparse, parse_qs
import json
import traceback
from machine_learning.services import classification_service
from machine_learning.services import regression_service
from machine_learning.preprocessing.parser import load_data
from machine_learning.models import classification, regression


@csrf_exempt
def offline_training(request):
    try:
        if request.method == 'GET':
            features, classification_target, regression_target = load_data(features_type='selected')
            response = {'classification': '', 'regression': ''}
            response['classification'] = classification.train(features, classification_target)
            response['regression'] = regression.train(features, regression_target)
            return JsonResponse(response)
    except Exception:
        traceback.print_tb(e.__traceback__)
        return JsonResponse({'mesage': 'invalid request'})


@csrf_exempt
def online_training_testing(request):
    try:
        if request.method == 'POST':
            path = request.get_full_path()
            type = parse_qs(urlparse(path).query)['type'][0]
            mode = parse_qs(urlparse(path).query)['mode'][0]
            data = json.loads(request.body)
            response = {}

            if type == 'classification':
                response = classification_service.run(mode, data)

            elif type == 'regression':
                response = regression_service.run(mode, data)

            return JsonResponse(response)

    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return JsonResponse({'mesage': 'invalid request'})
