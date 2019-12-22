import matplotlib.pyplot as plt
import numpy
#y-axis uni, x-axis resp
heatmap90 = [[200, 200, 500, 300, 700],
            [100, -300, 300, 800, 700],
            [-300, -100, 400, 500, 800],
            [-300, -500, -200, 200, 0],
            [-500, -300, -500, -500, -600]]



heatmap100 = [[400, 600, 100, 100, 100],
            [1600, 2200, 1900, 1000, 2700],
            [3300, 3500, 4200, 4700, 4800],
            [3700, 4700, 4500, 5900, 6100],
            [-500, 2600, 300, 900, 1200]]

xLabel = [0.1, 0.3, 0.5, 0.7, 0.9]
yLabel = [0.1, 0.3, 0.5, 0.7, 0.9]

heatmap90 = numpy.array(heatmap90)
heatmap100 = numpy.array(heatmap100)

heatmap90 = heatmap90/10000
heatmap100 = heatmap100/10000

plt.imshow(heatmap90, cmap = 'Reds')

plt.xticks(range(5), xLabel)
plt.yticks(range(5), yLabel)

plt.xlabel('responsiveness')
plt.ylabel('uniqueness')
plt.colorbar()
plt.show()


plt.imshow(heatmap100, cmap = 'Reds')
plt.xticks(range(5), xLabel)
plt.yticks(range(5), yLabel)
plt.xlabel('responsiveness')
plt.ylabel('uniqueness')
plt.colorbar()
plt.show()