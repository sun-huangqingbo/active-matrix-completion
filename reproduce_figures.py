import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# fig2a = pd.read_excel(r'C:\Users\57539\Desktop\AL manuscript wrap up\charts\uni0.3res0.1new test (threshold)32ptps.xlsx', header = None)
# fig2a_data = np.array(fig2a.values[:4, :100])
# np.save(r'C:\Users\57539\Desktop\AL manuscript wrap up\processed data & code\processed data\u0.3r0.0.1_adjusted.npy', fig2a_data)

# fig2b = pd.read_excel(r'C:\Users\57539\Desktop\AL manuscript wrap up\charts\uni0.3res0.1new test 32ptps.xlsx', header = None)
# fig2b_data = np.array(fig2b.values[:4, :100])
# np.save(r'C:\Users\57539\Desktop\AL manuscript wrap up\processed data & code\processed data\u0.3r0.0.1_no_adjustment.npy', fig2b_data)


# real_experiment_accuracy = pd.read_excel(r'C:\Users\57539\Desktop\AL manuscript wrap up\charts\real_experiment_accu(threshold).xlsx')
# real_experiment_vec_data = np.array(real_experiment_accuracy.values[:6, :100])
# np.save(r'C:\Users\57539\Desktop\AL manuscript wrap up\processed data & code\processed data\real_experiment_accuracy', real_experiment_vec_data)

# figure 2

fig2a = np.load(r'processed data\u0.3r0.0.1_adjusted.npy')
plt.figure(1, figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(fig2a[0], '-', c = 'darkorange')
plt.plot(fig2a[1], '--',c = 'deepskyblue')
plt.plot(fig2a[2], '-', c = 'limegreen')
plt.plot(fig2a[3], '--',c = 'brown')
plt.xlim(0, 100)
plt.ylim(0.7, 1)
plt.xlabel('Batch')
plt.ylabel("Accuracy")

fig2b = np.load(r'processed data\u0.3r0.0.1_no_adjustment.npy')
plt.subplot(1, 2, 2)
plt.plot(fig2b[0], '-', c = 'darkorange')
plt.plot(fig2b[1], '--',c = 'deepskyblue')
plt.plot(fig2b[2], '-', c = 'limegreen')
plt.plot(fig2b[3], '--',c = 'brown')
plt.xlim(0, 100)
plt.ylim(0.7, 1)
plt.xlabel('Batch')
plt.ylabel("Accuracy")
plt.suptitle('Figure2, a, b')
# plt.show()

# figure 3

fig3a = np.load(r'processed data\u0.2r0.2_adjusted.npy', allow_pickle=True).astype('f')
fig3b = np.load(r'processed data\u0.2r0.9_adjusted.npy', allow_pickle=True).astype('f')
fig3c = np.load(r'processed data\u0.6r0.2_adjusted.npy', allow_pickle=True).astype('f')
fig3d = np.load(r'processed data\u0.6r0.9_adjusted.npy', allow_pickle=True).astype('f')
plt.figure(2, figsize=(16, 8))
plt.subplot(2, 2, 1)
plt.plot(fig3a[0], '-', c = 'darkorange')
plt.plot(fig3a[1], '-',c = 'deepskyblue')
plt.plot(fig3a[2], '-', c = 'limegreen')
plt.plot(fig3a[3], '--',c = 'brown')
plt.xlim(0, 100)
plt.ylim(0.6, 1)
plt.xlabel('Batch')
plt.ylabel("Accuracy")

plt.subplot(2, 2, 2)
plt.plot(fig3b[0], '-', c = 'darkorange')
plt.plot(fig3b[1], '-',c = 'deepskyblue')
plt.plot(fig3b[2], '-', c = 'limegreen')
plt.plot(fig3b[3], '--',c = 'brown')
plt.xlim(0, 100)
plt.ylim(0, 1)
plt.xlabel('Batch')
plt.ylabel("Accuracy")

plt.subplot(2, 2, 3)
plt.plot(fig3c[0], '-', c = 'darkorange')
plt.plot(fig3c[1], '-',c = 'deepskyblue')
plt.plot(fig3c[2], '-', c = 'limegreen')
plt.plot(fig3c[3], '--',c = 'brown')
plt.xlim(0, 100)
plt.ylim(0.4, 1)
plt.xlabel('Batch')
plt.ylabel("Accuracy")

plt.subplot(2, 2, 4)
plt.plot(fig3d[0], '-', c = 'darkorange')
plt.plot(fig3d[1], '-',c = 'deepskyblue')
plt.plot(fig3d[2], '-', c = 'limegreen')
plt.plot(fig3d[3], '--',c = 'brown')
plt.xlim(0, 100)
plt.ylim(0, 1)
plt.xlabel('Batch')
plt.ylabel("Accuracy")
plt.suptitle('Figure3, a-d')


# figure 4
hybrid100 = np.load(r'processed data\hybrid_adjustment100.npy')/10000
entropy100 = np.load(r'processed data\entropy_adjustment100.npy')/10000
hybrid90 = np.load(r'processed data\hybrid_adjustment90.npy')/10000
random100 = np.load(r'processed data\random_adjustment100.npy')/10000
random90 = np.load(r'processed data\random_adjustment90.npy')/10000

plt.figure(3, figsize=(6, 6))
he100 = entropy100 - hybrid100
cmap = plt.cm.coolwarm
annotate = he100
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(he100, cmap=cmap, center=0, annot = annotate,
            square=True, linewidths=.5, cbar = False, vmin = -0.1, vmax = 0.8, fmt=".0%")
plt.xticks(np.arange(0, 9)+0.5, ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'] )
plt.yticks(np.arange(0, 9)+0.5, ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'], rotation=0)
plt.xlabel('Responsive')
plt.ylabel('Unique')
plt.title('Figure 4')

# figure 6
plt.figure(4,figsize=(14, 14))
plt.subplot(2, 2, 1)
hr90ad = random90 - hybrid90
cmap = plt.cm.coolwarm
annotate = hr90ad
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(hr90ad, cmap=cmap, center=0, annot = annotate,
            square=True, linewidths=.5, cbar = False, vmin = -0.1, vmax = 0.8, fmt=".0%")
plt.xticks(np.arange(0, 9)+0.5, ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'] )
plt.yticks(np.arange(0, 9)+0.5, ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'], rotation=0)
plt.xlabel('Responsive')
plt.ylabel('Unique')

plt.subplot(2, 2, 2)
hr100ad = random100 - hybrid100
cmap = plt.cm.coolwarm
annotate = hr100ad
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(hr100ad, cmap=cmap, center=0, annot = annotate,
            square=True, linewidths=.5, cbar = False, vmin = -0.1, vmax = 0.8, fmt=".0%")
plt.xticks(np.arange(0, 9)+0.5, ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'] )
plt.yticks(np.arange(0, 9)+0.5, ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'], rotation=0)
plt.xlabel('Responsive')
plt.ylabel('Unique')

hybrid100noad = np.load(r'processed data\hybrid100.npy')/10000
random100noad = np.load(r'processed data\random100.npy')/10000
hybrid90noad = np.load(r'processed data\hybrid90.npy')/10000
random90noad = np.load(r'processed data\random90.npy')/10000

plt.subplot(2, 2, 3)
hr90 = random90noad - hybrid90noad
cmap = plt.cm.coolwarm
annotate = hr90
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(hr90, cmap=cmap, center=0, annot = annotate,
            square=True, linewidths=.5, cbar = False, vmin = -0.1, vmax = 0.8, fmt=".0%")
plt.xticks(np.arange(0, 9)+0.5, ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'] )
plt.yticks(np.arange(0, 9)+0.5, ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'], rotation=0)
plt.xlabel('Responsive')
plt.ylabel('Unique')

plt.subplot(2, 2, 4)
hr100 = random100noad - hybrid100noad
cmap = plt.cm.coolwarm
annotate = hr100
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(hr100, cmap=cmap, center=0, annot = annotate,
            square=True, linewidths=.5, cbar = False, vmin = -0.1, vmax = 0.8, fmt=".0%")
plt.xticks(np.arange(0, 9)+0.5, ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'] )
plt.yticks(np.arange(0, 9)+0.5, ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'], rotation=0)
plt.xlabel('Responsive')
plt.ylabel('Unique')
plt.suptitle('Figure 6, a-d')


# Figure 8
accu = np.load(r'processed data\real_experiment_accuracy.npy')
vec = np.load(r'processed data\real_experiment_vec.npy')
plt.figure(5, figsize=(10, 16))
plt.subplot(3, 2, 1)
plt.plot(accu[0], '-', c='darkorange')
plt.plot(accu[1], '-', c='deepskyblue')
plt.plot(accu[2], '-', c='limegreen')
plt.plot(accu[3], '--', c='brown')
plt.plot(accu[4], '-', c='indigo')
plt.plot(accu[5], '--', c='darkgreen')
plt.xlim(0, 100)
plt.ylim(0.2, 1)
plt.xlabel('Batch')
plt.ylabel("Accuracy")

plt.subplot(3, 2, 2)
plt.plot(vec[0], '-', c='darkorange')
plt.plot(vec[5], '-', c='deepskyblue')
plt.plot(vec[10], '-', c='limegreen')
plt.plot(vec[15], '--', c='brown')
plt.plot(vec[20], '-', c='indigo')
plt.plot(vec[25], '--', c='darkgreen')
plt.xlim(0, 100)
plt.yticks(np.arange(0, 1.1, 0.1), ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'], rotation=0)
plt.xlabel('Batch')
plt.ylabel("Percentage")
i = 1
plt.subplot(3, 2, 3)
plt.plot(vec[0+ i], '-', c='darkorange')
plt.plot(vec[5 + i], '-', c='deepskyblue')
plt.plot(vec[10 + i], '-', c='limegreen')
plt.plot(vec[15 + i], '--', c='brown')
plt.plot(vec[20 + i], '-', c='indigo')
plt.plot(vec[25 + i], '--', c='darkgreen')
plt.xlim(0, 100)
plt.yticks(np.arange(0, 1.1, 0.1), ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'], rotation=0)
plt.xlabel('Batch')
plt.ylabel("Percentage")
i = 2
plt.subplot(3, 2, 4)
plt.plot(vec[0+ i], '-', c='darkorange')
plt.plot(vec[5 + i], '-', c='deepskyblue')
plt.plot(vec[10 + i], '-', c='limegreen')
plt.plot(vec[15 + i], '--', c='brown')
plt.plot(vec[20 + i], '-', c='indigo')
plt.plot(vec[25 + i], '--', c='darkgreen')
plt.xlim(0, 100)
plt.yticks(np.arange(0, 1.1, 0.1), ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'], rotation=0)
plt.xlabel('Batch')
plt.ylabel("Percentage")
i = 3
plt.subplot(3, 2, 5)
plt.plot(vec[0+ i], '-', c='darkorange')
plt.plot(vec[5 + i], '-', c='deepskyblue')
plt.plot(vec[10 + i], '-', c='limegreen')
plt.plot(vec[15 + i], '--', c='brown')
plt.plot(vec[20 + i], '-', c='indigo')
plt.plot(vec[25 + i], '--', c='darkgreen')
plt.xlim(0, 100)
plt.yticks(np.arange(0, 1.1, 0.1), ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'], rotation=0)
plt.xlabel('Batch')
plt.ylabel("Percentage")
i = 4
plt.subplot(3, 2, 6)
plt.plot(vec[0+ i], '-', c='darkorange')
plt.plot(vec[5 + i], '-', c='deepskyblue')
plt.plot(vec[10 + i], '-', c='limegreen')
plt.plot(vec[15 + i], '--', c='brown')
plt.plot(vec[20 + i], '-', c='indigo')
plt.plot(vec[25 + i], '--', c='darkgreen')
plt.xlim(0, 100)
plt.yticks(np.arange(0, 1.1, 0.1), ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'], rotation=0)
plt.xlabel('Batch')
plt.ylabel("Percentage")
plt.suptitle('Figure 8, a-f')

plt.show()