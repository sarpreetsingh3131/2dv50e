import json
import traceback
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from urllib.parse import urlparse, parse_qs
from machine_learner.models import classification, regression


@csrf_exempt
def training_testing(request):
    try:
        if request.method == 'POST':
            path = request.get_full_path()
            task_type = parse_qs(urlparse(path).query)['task-type'][0]
            mode = parse_qs(urlparse(path).query)['mode'][0]
            dataset = json.loads(request.body)
            response = {}
            if mode == 'comparison':
                response = save_data(dataset)
            elif task_type == 'classification':
                if mode == 'training':
                    response = classification.training(
                        dataset['features'], dataset['target'])
                elif mode == 'testing':
                    response = classification.testing(dataset['features'])
            elif task_type == 'regression':
                if mode == 'training':
                    response = regression.training(
                        dataset['features'], dataset['target'])
                elif mode == 'testing':
                    response = regression.testing(dataset['features'])
            return JsonResponse(response)
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return JsonResponse({'mesage': 'invalid request'})


def save_data(data):
    try:
        file_data = json.load(
            open('machine_learner/collected_data/selected_adaptation_options.json'))
    except Exception:
        file_data = []

    file_data.append({
        'packetLoss': data['packetLoss'],
        'energyConsumption': data['energyConsumption'],
        'classification': data['classification'],
        'regression': data['regression']
    })

    with open('machine_learner/collected_data/selected_adaptation_options.json', 'w') as f:
        json.dump(file_data, f, indent=4)

    return {'message': 'successful'}
