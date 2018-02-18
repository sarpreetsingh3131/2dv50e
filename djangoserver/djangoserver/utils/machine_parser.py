def parse_machine_data(adaptations, is_training):
    features = []
    labels = []

    for index, adaptation in enumerate(adaptations):
        features.append([])

        # for mote in adaptation['motes']:
        #    features[index].append(mote['moteId'])

        # for mote in adaptation['motes']:
        #    features[index].append(mote['load'])

        for mote in adaptation['motes']:
            features[index].append(mote['energyLevel'])

        # for mote in adaptation['motes']:
        #    features[index].append(mote['parents'])

        # for mote in adaptation['motes']:
        #    features[index].append(mote['queueSize'])

        for mote in adaptation['motes']:
            features[index].append(mote['source'])

        for mote in adaptation['motes']:
            features[index].append(mote['destination'])

        # for mote in adaptation['motes']:
        #    features[index].append(mote['powerSetting'])

        for mote in adaptation['motes']:
            features[index].append(mote['distribution'])

        for snr in adaptation['linksSNR']:
            features[index].append(snr['SNR'])

        for traffic in adaptation['motesTraffic']:
            features[index].append(traffic['traffic'])

        if is_training:
            loss = adaptation['packetLoss']
            if loss <= 10:
                labels.append(0)
            else:
                labels.append(1)

    print('features =', len(features), ', lables =', len(labels), labels)
    return {'features': features, 'labels': labels}
