def parse_training_adaptations(data):
    return parse(data['adaptations'], data['environment'], is_training=True)


def parse_testing_adaptations(data):
    return parse(data['adaptations'], data['environment'], is_training=False)


def parse(adaptations, environment, is_training):
    data = []
    for adaptation_index, adaptation_id in enumerate(adaptations):
        data.append({'adaptation_id': adaptation_id, 'motes': []})
        parse_motes(adaptation_index, adaptation_id, adaptations, data)
        parse_environment(adaptation_index, environment, data)

        if is_training:
            parse_verification_results(adaptation_index, adaptation_id, adaptations, data)

    return data


def parse_environment(adaptation_index, environment, data):
    for attribute in environment:
        data[adaptation_index][attribute] = environment[attribute]


def parse_verification_results(adaptation_index, adaptation_id, adaptations, data):
    for attribute in adaptations[adaptation_id]['verificationResults']:
        data[adaptation_index][attribute] = adaptations[adaptation_id]['verificationResults'][attribute]['value']['value']


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
