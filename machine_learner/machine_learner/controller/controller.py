import json
import os
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
            cycle = parse_qs(urlparse(path).query)['cycle'][0]

            # Clear the models/output at the first adaptation cycle
            if int(cycle) == 1:
                if (mode == 'comparison') or (mode == 'mladjustment'):
                    # Remove all the collected data files before saving the first time
                    # The collected data files in question are .txt and .json files
                    dir_path = os.path.join('machine_learner', 'collected_data')
                    deleteFilesWithExt(dir_path, ['.txt', '.json'])
                elif (mode == 'training') and ((task_type == 'classification') or (task_type == 'regression')):
                    # Remove all the trained models for the specified mode before they are trained for the first time
                    # This is done by searching all the model files and deleting them (in the respective subdirectory per task_type)
                    dir_path = os.path.join('machine_learner', 'trained_models', task_type)
                    deleteFilesWithExt(dir_path, ['.pkl'])

            dataset = json.loads(request.body)
            response = {}
            if mode == 'comparison':
                response = save_data(dataset)
            elif mode == 'mladjustment':
                response = save_mlAdjustmentData(dataset)
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
        else:
            return JsonResponse({'message': 'only POST requests are allowed'})
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return JsonResponse({'message': 'invalid request'})



def deleteFilesWithExt(dir_path, extensions):
    '''
        Deletes all files with an extension provided in extensions under a given path.
    '''
    files = os.listdir(dir_path)
    for item in files:
        for extension in extensions:
            if item.endswith(extension):
                os.remove(os.path.join(dir_path, item))



def save_mlAdjustmentData(data):
    outputPath = os.path.join('machine_learner', 'collected_data', 'overall_adaptation_options.json')
    try:
        overall_file = json.load(open(outputPath))
    except Exception:
        overall_file = []

    overall_file.append({'adaptationOptions' : []})

    for i in range(len(data['packetLoss'])):
        overall_file[len(overall_file)-1]['adaptationOptions'].append({
            'adaptationOption' : i,
            'packetLoss' : data['packetLoss'][i],
            'energyConsumption' : data['energyConsumption'][i], 
            'classificationBefore' : data['classificationBefore'][i],
            'regressionBefore' : data['regressionBefore'][i],
            'classificationAfter' : data['classificationAfter'][i],
            'regressionAfter' : data['regressionAfter'][i]
        })
        
    with open(outputPath, 'w') as f:
        json.dump(overall_file, f, indent=4)
        f.close()

    return {'message': 'successful'}


def save_data(data):
    outputPath = os.path.join('machine_learner', 'collected_data', 'selected_adaptation_options.json')
    try:
        file_data = json.load(
            open(outputPath))
    except Exception:
        file_data = []

    file_data.append({
        'packetLoss': data['packetLoss'],
        'energyConsumption': data['energyConsumption'],
        'classification': data['classification'],
        'regression': data['regression']
    })

    with open(outputPath, 'w') as f:
        json.dump(file_data, f, indent=4)

    return {'message': 'successful'}
