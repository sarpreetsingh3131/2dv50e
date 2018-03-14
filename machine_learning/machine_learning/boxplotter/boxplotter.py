from matplotlib import pyplot as plt

DIR_PATH = 'machine_learning/results/'

fig_index = 0
plt.figure(figsize=(15, 10))

for title in ['Energy Consumption', 'Packet Loss', 'Adaptation Space', 'Time (sec)']:
    data = {'activform': [], 'classification_1': [], 'regression_1': []}

    for file_name in ['activform', 'classification_1', 'regression_1']:
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

            elif title == 'Time (sec)' and len(content) == 4:
                data[file_name].append((float(content[3]) - float(content[1])) / 1000.0)

    plt.subplot(2, 2, fig_index + 1)

    plt.boxplot([data[x] for x in ['activform', 'classification_1', 'regression_1']], positions=[1, 2, 3],
                labels=['ActivFORM', 'Classification', 'Regression'], widths=.5)

    plt.ylabel(title)
    fig_index += 1

plt.suptitle("Comparison")
plt.savefig(DIR_PATH + 'Comparison.pdf')
print('graphs saved')
