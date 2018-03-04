import json
import traceback

DIR_PATH = 'machine_learning/training_data/'


def parse(type):
    features = []
    target = []

    try:
        # read from json files (faster than parsing every time)
        for index, file_name in enumerate(['features.json', type + '_target.json']):
            with open(DIR_PATH + file_name) as json_data:
                if index == 0:
                    features = json.load(json_data)
                else:
                    target = json.load(json_data)

    except Exception:
        # parse data from txt file, then save into json file and return it
        count = 0
        for file_name in ['data1.txt', 'data2.txt', 'data3.txt', 'data4.txt', 'data5.txt',  'data6.txt',  'data7.txt']:
            with open(DIR_PATH + file_name) as f:
                file_data = f.readlines()

            file_data = [x.strip() for x in file_data]  # remove all '\n'

            for cycle in file_data:
                cycle = json.loads(cycle)
                cycle_features = []
                cycle_target = []

                for index, adaptation in enumerate(cycle['adaptations']):
                    cycle_features.append([])

                    for mote in adaptation:
                        for link in mote['links']:
                            cycle_features[index].append(int(str(link['SNR']).split('.')[0]))

                    for mote in adaptation:
                        for link in mote['links']:
                            cycle_features[index].append(link['distribution'])

                    for mote in adaptation:
                        for link in mote['links']:
                            cycle_features[index].append(link['power'])

                    for mote in adaptation:
                        cycle_features[index].append(mote['dataProbability'])

                for index, result in enumerate(cycle['result']):
                    for item in result:
                        if type == 'classification':
                            if float(item['packetLoss']) < 0.10:
                                cycle_target.append(1)
                            else:
                                cycle_target.append(0)
                        else:
                            cycle_target.append(float(item['packetLoss']))

                for index, feature in enumerate(cycle_features):
                    features.append(feature)
                    target.append(cycle_target[index])

        try:
            for index, file_name in enumerate(['features.json', type + '_target.json']):
                with open(DIR_PATH + file_name, 'w') as f:
                    if index == 0:
                        json.dump(features, f)
                    else:
                        json.dump(target, f)
            print('parsed data saved in json file')
        except Exception as e:
            traceback.print_tb(e.__traceback__)
            print('cannot save parsed data into json file')

    return {'features': features, 'target': target}
