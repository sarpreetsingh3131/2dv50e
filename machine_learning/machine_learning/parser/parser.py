import numpy as np


def parse_data(data, target_type=None):
    features = []
    target = []
    adaptations = parse_adaptations(data['adaptations'], data['environment'])

    for index, adaptation in enumerate(adaptations):
        features.append([])

        for snr in adaptation['linksSNR']:
            features[index].append(snr['SNR'])

        for mote in adaptation['motes']:
            mote_id = mote['moteId']
            if mote_id == 7 or mote_id == 10 or mote_id == 12:
                features[index].append(mote['distribution'])

        for traffic in adaptation['motesTraffic']:
            mote_id = traffic['moteId']
            if mote_id == 10 or mote_id == 12:
                features[index].append(traffic['traffic'])

        if target_type == 'classification':
            if adaptation['packet_loss'] < 10:
                target.append(1)
            else:
                target.append(0)

        elif target_type == 'regression':
            target.append(adaptation['packet_loss'])

    return np.array(features, dtype='float64'), np.array(target, dtype='float64')


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
