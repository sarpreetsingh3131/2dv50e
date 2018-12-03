import matplotlib.pyplot as plt

fig = plt.figure()
kf = 10
plt.bar(range(1,kf+1),range(1,kf+1), label="Scores over all folds")
plt.xticks(range(1,kf+1))
plt.hlines(5, 0 , kf+1, 'r', label="Mean")
plt.show()