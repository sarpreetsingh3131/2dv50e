from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from urllib.parse import urlparse, parse_qs
import json
import traceback
from machine_learning.services import classification_service
from machine_learning.services import regression_service


@csrf_exempt
def handle_request(request):
    try:
        path = request.get_full_path()
        type = parse_qs(urlparse(path).query)['type'][0]
        mode = parse_qs(urlparse(path).query)['mode'][0]
        model_name = parse_qs(urlparse(path).query)['model_name'][0]
        data = {}
        response = {}

        if request.method == 'POST':
            data = json.loads(request.body)

        if type == 'classification':
            response = classification_service.run(mode, model_name, data)

        elif type == 'regression':
            response = regression_service.run(mode, model_name, data)

        return JsonResponse(response)

    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return JsonResponse({'mesage': 'invalid request'})
