from matplotlib import pyplot as plt
import numpy as np

DIR_PATH = 'machine_learning/results/'


data = {'classification': [], 'regression': [], 'activform': []}

with open(DIR_PATH + 'data/online_learning.txt') as f:
    file_data = f.readlines()

file_data = [x.strip() for x in file_data]

for line in file_data:
    content = line.split(';')

    if len(content) == 4:
        data['classification'].append(int(content[1]))
        data['regression'].append(int(content[2]))
        data['activform'].append(int(content[3]))



plt.figure(figsize=(10, 8))
plt.xlim(-1, 80)
x_axis = np.arange(1, 77, 1);

plt.plot(x_axis, data['classification'], label='Classification')
plt.plot(x_axis, data['regression'], label='Regression')
plt.plot(x_axis, data['activform'], label='ActivFORM')

plt.legend(loc='upper right')
plt.xlabel("Cycles")
plt.ylabel("Adaptation Space")

plt.suptitle('Online Incremental Learning')
plt.savefig(DIR_PATH + 'Online Learning.pdf')
print('graph saved')
