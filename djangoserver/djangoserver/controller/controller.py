from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from urllib.parse import urlparse, parse_qs
import json
import djangoserver.models.classification as classification
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, AdaBoostClassifier, BaggingClassifier


@csrf_exempt
def handle(request):
    if request.method == 'POST':
        try:
            path = request.get_full_path()
            mode = parse_qs(urlparse(path).query)['mode'][0]
            clf_name = parse_qs(urlparse(path).query)['clf'][0]
            type = parse_qs(urlparse(path).query)['type'][0]
            data = json.loads(request.body)
            response = ''
            print(mode, clf_name, type)

            if type == 'classification':
                response = execute(mode, classification, classification_clfs[clf_name], clf_name, data)

            elif type == 'regression':
                pass

            elif type == 'clustering':
                pass

            return JsonResponse(response)

        except Exception:
            return JsonResponse({'mesage': 'invalid request'})


def execute(mode, model, clf, clf_name, data):
    if mode == 'training':
        return model.train_model(clf, clf_name, data)
    elif mode == 'testing':
        return model.test_model(clf, clf_name, data)


classification_clfs = {
    'random_forest': RandomForestClassifier(),
    'extra_tree': ExtraTreesClassifier(),
    'voting': VotingClassifier(estimators=[('random_forest', RandomForestClassifier())]),
    'ada_boost': AdaBoostClassifier(),
    'bagging': BaggingClassifier()
}
