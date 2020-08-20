import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
hybrid100 = np.load('processed data' + os.path.sep + 'hybrid100.npy')/10000
hybrid90 = np.load('processed data' + os.path.sep + 'hybrid90.npy')/10000
random100 = np.load('processed data' + os.path.sep + 'random100.npy')/10000
random90 = np.load('processed data' + os.path.sep + 'random90.npy')/10000

plt.figure(1,figsize=(10, 6))
plt.subplot(1, 2, 1)
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

plt.subplot(1, 2, 2)
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


plt.figure(2,figsize=(8, 6))
gm = np.load('processed data' + os.path.sep + 'gm90.npy')/10000

ibc = np.load('processed data' + os.path.sep + 'latest90.npy')/10000

si = np.load('processed data' + os.path.sep + 'si90.npy')/10000


x = np.arange(10)  # the label locations
width = 0.5  # the width of the bars

rects1 = plt.bar(x - 0.5*width, gm, 0.5*width, label='Naik et al. Active Model', color='g')
rects2 = plt.bar(x, si, 0.5*width, label='Chen et al. Active Model', color='b')
rects3 = plt.bar(x + 0.5*width, ibc, 0.5*width, label='Our Active Model - Hybrid Query', color='orange')
plt.xticks(x, ['u=0.4, r=0.2', 'u=0.4, r=0.4', 'u=0.4, r=0.6', 'u=0.4, r=0.8', 'u=0.4, r=1.0', 
'u=0.8, r=0.2', 'u=0.8, r=0.4', 'u=0.8, r=0.6', 'u=0.8, r=0.8', 'u=0.8, r=1.0'], rotation = 50)
plt.yticks(np.arange(0, 0.9, 0.1), ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%'], rotation=0)
plt.legend(loc='best')
plt.figure(3,figsize=(8, 6))
real_result = np.load('processed data' + os.path.sep + "image_results.npy")

plt.plot(real_result[0], c='orange', label = 'Naik et al. Active Model')
plt.plot(real_result[1], c='blue', label = 'Our Active Model - Hybrid Query')
plt.plot(real_result[2], c='g', label = 'Our Active Model - Least Score')
plt.plot(real_result[3], c='violet', label = 'Our Active Model - Entropy')
plt.plot(real_result[4], c='brown', label = 'Our Random Model')
plt.plot(real_result[5], c='goldenrod', label = 'Naik et al. Random Model')
plt.yticks(np.arange(0, 1.1, 0.2), ['0%', '20%', '40%', '60%', '80%', '100%'], rotation=0)
plt.legend(loc='best')
plt.show()