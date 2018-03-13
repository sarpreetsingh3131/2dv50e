import json
import numpy as np


def load_data(features_type):
    features = []
    classfication_target = []
    regression_target = []

    with open('machine_learning/preprocessing/data.txt') as f:
        file_data = f.readlines()

    file_data = [x.strip() for x in file_data]

    for cycle in file_data:
        cycle_features, cycle_classification_target, cycle_regression_target = parse_data(
            json.loads(cycle), features_type)

        for index in range(0, len(cycle_features)):
            features.append(cycle_features[index])
            classfication_target.append(cycle_classification_target[index])
            regression_target.append(cycle_regression_target[index])

    return np.array(
        features, dtype='float64'), np.array(
        classfication_target, dtype='float64'), np.array(
        regression_target, dtype='float64')


def parse_data(data, features_type):
    features = []
    classfication_target = []
    regression_target = []
    adaptations = parse_adaptations(data['adaptations'], data['environment'])

    for index, adaptation in enumerate(adaptations):
        features.append([])

        for snr in adaptation['linksSNR']:
            features[index].append(snr['SNR'])

        if features_type == 'all':
            for mote in adaptation['motes']:
                features[index].append(mote['distribution'])

            for traffic in adaptation['motesTraffic']:
                features[index].append(traffic['traffic'])

        elif features_type == 'selected':
            for mote in adaptation['motes']:
                mote_id = mote['moteId']
                if mote_id == 7 or mote_id == 10 or mote_id == 12:
                    features[index].append(mote['distribution'])

            for traffic in adaptation['motesTraffic']:
                mote_id = traffic['moteId']
                if mote_id == 10 or mote_id == 12:
                    features[index].append(traffic['traffic'])

        if adaptation['packet_loss'] < 10:
            classfication_target.append(1)
        else:
            classfication_target.append(0)

        regression_target.append(adaptation['packet_loss'])

    return np.array(
        features, dtype='float64'), np.array(
        classfication_target, dtype='float64'), np.array(
        regression_target, dtype='float64')


def parse_adaptations(adaptations, environment):
    data = []
    for adaptation_index, adaptation_id in enumerate(adaptations):
        data.append({'adaptation_id': adaptation_id, 'motes': [], 'packet_loss': 0, 'energy_consumption': 0})
        parse_motes(adaptation_index, adaptation_id, adaptations, data)
        parse_environment(adaptation_index, environment, data)
        parse_verification_results(adaptation_index, adaptation_id, adaptations, data)
    return data


def parse_verification_results(adaptation_index, adaptation_id, adaptations, data):
    packet_loss = adaptations[adaptation_id]['verificationResults']['packetLoss']['value']['value']
    energy_consumption = adaptations[adaptation_id]['verificationResults']['energyConsumption']['value']['value']
    data[adaptation_index]['packet_loss'] = packet_loss
    data[adaptation_index]['energy_consumption'] = energy_consumption


def parse_environment(adaptation_index, environment, data):
    for attribute in environment:
        data[adaptation_index][attribute] = environment[attribute]


def parse_motes(adaptation_index, adaptation_id, adaptations, data):
    mote_index = 0
    for mote_id in adaptations[adaptation_id]['deltaIoT']['motes']:
        data[adaptation_index]['motes'].append({})

        for mote_attribute in adaptations[adaptation_id]['deltaIoT']['motes'][mote_id]:
            element = adaptations[adaptation_id]['deltaIoT']['motes'][mote_id][mote_attribute]

            if mote_attribute == 'links':
                for link_index, link_id in enumerate(element):
                    for link_attribute in element[link_id]:
                        link_value = element[link_id][link_attribute]['value']['value']
                        data[adaptation_index]['motes'][mote_index][link_attribute] = link_value

                    if data[adaptation_index]['motes'][mote_index]['parents'] == 1:
                        break
                    elif link_index == 0:
                        mote = data[adaptation_index]['motes'][mote_index]
                        data[adaptation_index]['motes'].append({
                            'moteId': mote['moteId'],
                            'load': mote['load'],
                            'energyLevel': mote['energyLevel'],
                            'parents': mote['parents'],
                            'queueSize': mote['queueSize']
                        })
                        mote_index += 1

            else:
                data[adaptation_index]['motes'][mote_index][mote_attribute] = element['value']['value']

        mote_index += 1
