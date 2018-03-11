from matplotlib import pyplot as plt
import traceback

DIR_PATH = 'machine_learning/results/'


def add_data(title, file_data, data, identifier):
    file_data = [x.strip() for x in file_data]

    for line in file_data:
        content = line.split(';')

        if title == 'Energy Consumption' and len(content) == 2:
            data[identifier].append(float(content[1]))

        elif title == 'Packet Loss' and len(content) == 2:
            data[identifier].append(float(content[0]))

        elif title == 'Adaptation Space' and len(content) == 4:
            data[identifier].append(float(content[2]))

        elif title == 'Time (sec)' and len(content) == 4:
            data[identifier].append((float(content[3]) - float(content[1])) / 1000.0)


def create_boxplot(title, data, identifiers, positions, labels, x_label, y_label, fig_name):
    fig, ax = plt.subplots()
    ax.set_title(title)

    boxplot_dict = ax.boxplot([data[x] for x in identifiers], positions=positions,
                              labels=labels, patch_artist=True, widths=.5)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    fig.savefig(DIR_PATH + fig_name + '.pdf')
    plt.close(fig)
    print('graphs saved')


def draw_boxplots():
    for title in ['Energy Consumption', 'Packet Loss', 'Adaptation Space', 'Time (sec)']:

        for result_type in['Classification', 'Regression']:
            data = {'_1': [], '_2': [], '_3': [], '_4': [], '_5': [], '_6': [], '_7': [], '_8': [], '_9': [], '_10': []}

            for file_number in ['_1', '_2', '_3', '_4', '_5', '_6', '_7', '_8', '_9', '_10']:
                with open(DIR_PATH + 'data/' + str(result_type).lower() + file_number + '.txt') as f:
                    file_data = f.readlines()

                add_data(title, file_data, data, file_number)

            create_boxplot(result_type + ' Online Learning', data,
                           ['_1', '_2', '_3', '_4', '_5', '_6', '_7', '_8', '_9', '_10'],
                           [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           'Testing Cycles', title, result_type + ' ' + title)

        data = {'activform': [], 'classification_10': [], 'regression_10': []}

        for file_name in ['activform', 'classification_10', 'regression_10']:
            with open(DIR_PATH + 'data/' + file_name + '.txt') as f:
                file_data = f.readlines()

            add_data(title, file_data, data, file_name)

        create_boxplot('Comparison', data, ['activform', 'classification_10', 'regression_10'],
                       [1, 2, 3], ['ActivFORM', 'Classification', 'Regression'], '', title, title + ' Comparison')


draw_boxplots()
