import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# y: u, x: r


ST90 = np.load(r'Active_Learning\heatmap\hybrid90.npy')/10000
ST100 = np.load(r'Active_Learning\heatmap\hybrid100.npy')/10000
RT90 = np.load(r'Active_Learning\heatmap\random90.npy')/10000
RT100 = np.load(r'Active_Learning\heatmap\random100.npy')/10000
M90 = RT90 - ST90

M100 = RT100 - ST100


def make_heatmap(m, sig, center):
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(6, 6))

    # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(220, 10, as_cmap=True, reverse = sig)
    if sig:
        cmap = plt.cm.coolwarm_r
    else:
        cmap = plt.cm.coolwarm

    annotate = m
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(m, cmap=cmap, center=center, annot = annotate,
                square=True, linewidths=.5, cbar = False, vmin = -0.1, vmax = 0.8, fmt=".0%")
    plt.xticks(np.arange(0, 9)+0.5, ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'] )
    plt.yticks(np.arange(0, 9)+0.5, ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'], rotation=0)
    plt.xlabel('Responsive')
    plt.ylabel('Unique')
    plt.show()
for m, sig, center in zip([ST90, ST100, RT90, RT100, M90, M100], [True, True, True, True, False, False], [0.5, 0.5, 0.5, 0.5, 0, 0]):
    make_heatmap(m, sig, center)
