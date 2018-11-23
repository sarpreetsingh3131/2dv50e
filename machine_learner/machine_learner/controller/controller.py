import json
import os
import traceback
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from urllib.parse import urlparse, parse_qs
from machine_learner.models import classification, regression, plLaClassification


@csrf_exempt
def training_testing(request):
    """ Function called when a request is received on localhost:8000 and responds apropriatly

    This function answers the httpRequest holding a json and a query from 
    activforms
    """
    try:
        # only post request
        if request.method == 'POST':

            # You get the full url from the request
            # scheme://netloc/path;parameters?query#fragment
            path = request.get_full_path()
            
            # You parse the url/path and only take the query part
            # You parse this query in string form
            # You take the value of the variable task-type
            task_type = parse_qs(urlparse(path).query)['task-type'][0]

            # Do the same to get the value of mode
            mode = parse_qs(urlparse(path).query)['mode'][0]

            cycle = int(parse_qs(urlparse(path).query)['cycle'][0])

            # Clear the models/output at the first adaptation cycle
<<<<<<< HEAD
            if cycle == 1:
=======
            # TODO: maybe make this optional, becuase
            # when we want to see how good ml can get,
            # it should be used on models who were trained
            # on tens of thousands of cases
            # an not the usual 30*216
            # We can do this buy training on a lot of data and than using those model.
            if int(cycle) == 1:
>>>>>>> latency
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

            # you load the received json 
            # Which is in the body of the post request
            # and load the string in to a json object
            # and assign it to dataset
            dataset = json.loads(request.body)

            #Initialise the response variable
            response = {}

            # Take the appropriate action depending on the
            # variables of the query
            if mode == 'comparison':

                #this only gets called 
                # at the end of the comparisson function of java
                # which gets called at each cycle for 
                # mode comparisson
                # This json has 4 keys, each with an array
                # classification: which holds the class prediction made 
                # for this cycle, which were send before this
                # regression: which holds the prediction made 
                # for this cycle, which were send before this
                # packetLoss: holds the results returned by 
                # activforms for the packetloss of this cycle
                # for every configuration. This is called
                # after the predictions of the machine learner
                # have been received.
                # The simulation of activforms is seen as
                # fact of what packetloss would have 
                # been in the next cycle
                # if you would
                # have applied that configuration
                # energyConsumption: same as packetLoss

                # This data is here to analyze after everything
                # has been run
                # The save data appends the data to the already existing data
                # of the previous cycle or creates a new file if its the first cycle
                # What if you run the program in comparisson 2 times?
                # This mode.comparisson should better be called 
                # mode.saveComparisson
                response = save_data(dataset)

            elif mode == 'mladjustment':
                response = save_mlAdjustmentData(dataset)

            # The next if statements are selfexplanatory

            #TODO: write the format of the returned json for every kind of response
            elif task_type == 'classification':
                
                if mode == 'training':
                    response = classification.training(
                        dataset['features'], dataset['target'], cycle)

                elif mode == 'testing':
                    response = classification.testing(dataset['features'])


            elif task_type == 'plLaClassification':
                
                if mode == 'training':
                    response = plLaClassification.training(
                        dataset['features'], dataset['target'])

                elif mode == 'testing':
                    response = plLaClassification.testing(dataset['features'])



            elif task_type == 'regression':

                if mode == 'training':
                    response = regression.training(
                        dataset['features'], dataset['target'], cycle)

                elif mode == 'testing':
                    response = regression.testing(dataset['features'])

            # You send the appropriate response back to the
            # process who requested
            # The json response was initialised empty
            # and has been changed to what it should be
            # except for of the mode was comparisson
            # in which case you send an empty json back
            # and the activforms does not look at that response
            # so it is no problem
            # It actually sends a message back it worked
            # for comparisson
            return JsonResponse(response)

        # If anything else then post request, send error message back
        else:
            return JsonResponse({'message': 'only POST requests are allowed'})

    # Something went seriously wrong.
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
    '''
        Saves the data retreived from the mladjustment mode in the simulator to a .json file.
    '''
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
    """ Appends or creates comparrison data

    This function appends the data of the next cycle of comparisson
    to the already existing data or creates a new file with the data
    if the file has not been created yet (or deleted)
    """

    # loads the existing data if any
    outputPath = os.path.join('machine_learner', 'collected_data', 'selected_adaptation_options.json')
    
    try:
    
        file_data = json.load(open(outputPath))
    
    # This exeption gets thrown when the file is not there
    # but maybe also if you have no read rights (maybe even write
    # or it is open in another program?)
    except Exception:
    
        file_data = []

    # append the new data to the existing data
    file_data.append({
        'packetLoss': data['packetLoss'],
        'energyConsumption': data['energyConsumption'],
        'classification': data['classification'],
        'regression': data['regression']
    })

    # with open opens the file en closes it after the block
    # it is equivalent to f = open(); do stuff; f.close()
    # This fails if you have not permission or it is open somewhere else
    # TODO: catch exception
    with open(outputPath, 'w') as f:
        json.dump(file_data, f, indent=4)

    return {'message': 'successful'}
