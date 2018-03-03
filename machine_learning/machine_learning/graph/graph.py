from matplotlib import pyplot as plt
import traceback

DATA_DIR_PATH = 'machine_learning/results/data/'
FIG_DIR_PATH = 'machine_learning/results/figures/'
COLORS = {'ActivFORM': 'orange', 'Regression': 'blue', 'Classification': 'green'}
CONTROLS = ['ActivFORM', 'Regression', 'Classification']


def draw():
    try:
        for index, title in enumerate(['Energy Consumption', 'Packet Loss', 'Adaptation Space', 'Time']):
            data = {
                'activform': [],
                'regression': [],
                'classification': []
            }

            for file_name in ['regression.txt', 'activform.txt', 'classification.txt']:
                with open(DATA_DIR_PATH + file_name) as f:
                    file_data = f.readlines()

                file_data = [x.strip() for x in file_data]

                for line in file_data:
                    content = line.split(';')
                    if title == 'Energy Consumption' and len(content) == 2:
                        add_result(file_name, data, float(content[1]))
                    elif title == 'Packet Loss' and len(content) == 2:
                        add_result(file_name, data, float(content[0]))
                    elif title == 'Adaptation Space' and len(content) == 4:
                        add_result(file_name, data, int(content[2]))
                    elif title == 'Time' and len(content) == 4:
                        add_result(file_name, data, (float(content[3]) - float(content[1])) / 1000.0)

            fig, ax = plt.subplots()
            ax.set_title(title)

            boxplot_dict = ax.boxplot(
                [data[x] for x in ['activform', 'regression', 'classification']],
                positions=[1, 2, 3],
                labels=CONTROLS,
                patch_artist=True,
                widths=.5)

            for index, box in enumerate(boxplot_dict['boxes']):
                lab = ax.get_xticklabels()[index].get_text()
                box.set_facecolor(COLORS[lab])

            fig.savefig(FIG_DIR_PATH + title)
            plt.close(fig)

        return {'message': 'graph saved'}
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return {'message': 'graph failed'}


def add_result(file_name, data, result):
    if file_name == 'regression.txt':
        data['regression'].append(result)
    elif file_name == 'activform.txt':
        data['activform'].append(result)
    else:
        data['classification'].append(result)
