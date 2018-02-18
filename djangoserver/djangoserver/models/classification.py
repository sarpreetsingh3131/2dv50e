from djangoserver.utils.models_db import save_model, get_saved_model
from djangoserver.utils.request_parser import parse_training_adaptations, parse_testing_adaptations
from djangoserver.utils.machine_parser import parse_machine_data
import traceback

path = 'djangoserver/djangoserver/resources/classification/'


def train_model(clf, clf_name, data):
    message = ''

    try:
        adaptations = parse_training_adaptations(data)
        machine_data = parse_machine_data(adaptations, is_training=True)
        model = get_saved_model(clf, path + clf_name + '.pkl')
        model.fit(machine_data['features'], machine_data['labels'])
        save_model(model, path + clf_name + '.pkl')
        message = clf_name + ' model trained and saved successfully'

    except Exception as e:
        traceback.print_tb(e.__traceback__)
        message = clf_name + ' model trained unsuccessfully '

    print(message)
    return {'message': message}


def test_model(clf, clf_name, data):
    message = ''
    predictions = []
    result = []

    try:
        adaptations = parse_testing_adaptations(data)
        machine_data = parse_machine_data(adaptations, is_training=False)
        model = get_saved_model(clf, path + clf_name + '.pkl')
        predictions = model.predict(machine_data['features'])
        message = clf_name + ' model tested and saved successfully'

    except Exception as e:
        traceback.print_tb(e.__traceback__)
        message = clf_name + ' model tested unsuccessfully'

    adaptation_space = 0

    for prediction in predictions:
        result.append(int(prediction))
        if prediction == 0:
            adaptation_space += 1

    print('predictions =', predictions, ' adaptation_space =', adaptation_space)
    return {'message': message, 'result': result, 'adaptation_space': adaptation_space}
