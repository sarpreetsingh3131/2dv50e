from matplotlib import pyplot as plt
import traceback

DATA_DIR_PATH = 'machine_learning/results/data/'
FIG_DIR_PATH = 'machine_learning/results/figures/'
#COLORS = {'ActivFORM': 'lightskyblue', 'Regression': 'lightgreen', 'Classification': 'lightyellow'}
COLORS = {'ActivFORM': 'lightskyblue', 'Classification': 'lightyellow'}
#CONTROLS = ['ActivFORM', 'Regression', 'Classification']
CONTROLS = ['ActivFORM', 'Classification']


def draw():
    try:
        for index, title in enumerate(['Energy Consumption', 'Packet Loss', 'Adaptation Space', 'Time']):
            #data = {'activform': [], 'regression': [], 'classification': []}
            data = {'activform': [], 'classification': []}

            for file_name in ['activform', 'classification']:
                with open(DATA_DIR_PATH + file_name + '.txt') as f:
                    file_data = f.readlines()

                file_data = [x.strip() for x in file_data]

                for line in file_data:
                    content = line.split(';')
                    if title == 'Energy Consumption' and len(content) == 2:
                        data[file_name].append(float(content[1]))  # add_result(file_name, data, float(content[1]))
                    elif title == 'Packet Loss' and len(content) == 2:
                        data[file_name].append(float(content[0]))  # add_result(file_name, data, float(content[0]))
                    elif title == 'Adaptation Space' and len(content) == 4:
                        data[file_name].append(float(content[2]))  # add_result(file_name, data, int(content[2]))
                    elif title == 'Time' and len(content) == 4:
                        data[file_name].append((float(content[3]) - float(content[1])) / 1000.0)
                        #add_result(file_name, data, (float(content[3]) - float(content[1])) / 1000.0)

            fig, ax = plt.subplots()
            ax.set_title(title)

            boxplot_dict = ax.boxplot([data[x] for x in ['activform', 'classification']],
                                      positions=[1, 2], labels=CONTROLS, patch_artist=True, widths=.5)

            for index, box in enumerate(boxplot_dict['boxes']):
                lab = ax.get_xticklabels()[index].get_text()
                box.set_facecolor(COLORS[lab])

            fig.savefig(FIG_DIR_PATH + title)
            plt.close(fig)

        print('graphs saved')
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        print('graphs failed')


draw()
