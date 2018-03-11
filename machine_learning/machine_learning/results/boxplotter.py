from matplotlib import pyplot as plt
import traceback

DIR_PATH = 'machine_learning/results/'


def create_boxplot_for_runs_result(result_type):
    for title in ['Energy Consumption', 'Packet Loss', 'Adaptation Space', 'Time']:
        data = {'_1': [], '_2': [], '_3': [], '_4': [], '_5': [], '_6': [], '_7': [], '_8': [], '_9': [], '_10': []}

        for file_number in ['_1', '_2', '_3', '_4', '_5', '_6', '_7', '_8', '_9', '_10']:
            with open(DIR_PATH + 'data/' + str(result_type).lower() + file_number + '.txt') as f:
                file_data = f.readlines()

            file_data = [x.strip() for x in file_data]

            for line in file_data:
                content = line.split(';')
                if title == 'Energy Consumption' and len(content) == 2:
                    data[file_number].append(float(content[1]))
                elif title == 'Packet Loss' and len(content) == 2:
                    data[file_number].append(float(content[0]))
                elif title == 'Adaptation Space' and len(content) == 4:
                    data[file_number].append(float(content[2]))
                elif title == 'Time' and len(content) == 4:
                    data[file_number].append((float(content[3]) - float(content[1])) / 1000.0)

        fig, ax = plt.subplots()
        ax.set_title(result_type + ' ' + title)

        boxplot_dict = ax.boxplot([data[x] for x in ['_1', '_2', '_3', '_4', '_5', '_6', '_7', '_8', '_9', '_10']],
                                  positions=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], patch_artist=True, widths=.5)

        ax.set_xlabel('Learning Cycle')
        ax.set_ylabel(title)

        fig.savefig(DIR_PATH + result_type + title + '.pdf')
        plt.close(fig)
        print('graphs saved')


def create_boxplot_for_final_result():
    colors = {'ActivFORM': 'lightskyblue', 'Classification': 'lightyellow', 'Regression': 'lightgreen'}

    for title in ['Energy Consumption', 'Packet Loss', 'Adaptation Space', 'Time']:
        data = {'activform': [], 'classification_10': [], 'regression_10': []}

        for file_name in ['activform', 'classification_10', 'regression_10']:
            with open(DIR_PATH + 'data/' + file_name + '.txt') as f:
                file_data = f.readlines()

            file_data = [x.strip() for x in file_data]

            for line in file_data:
                content = line.split(';')
                if title == 'Energy Consumption' and len(content) == 2:
                    data[file_name].append(float(content[1]))
                elif title == 'Packet Loss' and len(content) == 2:
                    data[file_name].append(float(content[0]))
                elif title == 'Adaptation Space' and len(content) == 4:
                    data[file_name].append(float(content[2]))
                elif title == 'Time' and len(content) == 4:
                    data[file_name].append((float(content[3]) - float(content[1])) / 1000.0)

        fig, ax = plt.subplots()
        ax.set_title(title)

        boxplot_dict = ax.boxplot([data[x] for x in['activform', 'classification_10', 'regression_10']],
                                  positions=[1, 2, 3], labels=['ActivFORM', 'Classification', 'Regression'],
                                  patch_artist=True, widths=.5)

        for index, box in enumerate(boxplot_dict['boxes']):
            lab = ax.get_xticklabels()[index].get_text()
            box.set_facecolor(colors[lab])

        fig.savefig(DIR_PATH + title + '.pdf')
        plt.close(fig)
        print('graphs saved')


create_boxplot_for_final_result()
create_boxplot_for_runs_result(result_type='Classification')
create_boxplot_for_runs_result(result_type='Regression')
