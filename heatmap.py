import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# y: u, x: r


softimpute90 = np.load(r'Active_Learning/feature90AL.npy')
softimpute100 = np.load(r'Active_Learning/feature100AL.npy')
onebit90 = np.load(r'Active_Learning/feature90RAND.npy')
onebit100 = np.load(r'Active_Learning/feature100RAND.npy')




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
    plt.xticks(np.arange(0, 4)+0.5, ['20%', '40%', '60%', '80%' ] )
    plt.yticks(np.arange(0, 4)+0.5, ['20%', '40%', '60%', '80%'], rotation=0)
    plt.xlabel('Responsive')
    plt.ylabel('Unique')
    plt.show()
for m, sig, center in zip([softimpute90, softimpute100, onebit90, onebit100], [True, True, True, True, True, True], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]):
    make_heatmap(m, sig, center)
