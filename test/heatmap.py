import matplotlib.pyplot as plt
import numpy
#x-axis uni, y-axis resp
heatmap90H = [[300, 300, 300, 200],
            [300, 0, 0, 300],
            [200, 700, 1200, 1100],
            [500, 1200, 200, 1600]]



heatmap100H = [[2500, 4000, 2600, 1200],
            [3600, 4300, 3200, 1200],
            [3800, 5000, 4000, 2200],
            [5700, 5000, 4800, 2500]]

xLabel = [0.2, 0.4, 0.6, 0.8]
yLabel = [0.2, 0.4, 0.6, 0.8]

heatmap90H = numpy.array(heatmap90H)
heatmap100H = numpy.array(heatmap100H)

heatmap90H = heatmap90H/10000
heatmap100H = heatmap100H/10000

plt.imshow(heatmap90H, cmap = 'Reds')

plt.xticks(range(4), xLabel)
plt.yticks(range(4), yLabel)

plt.ylabel('responsiveness')
plt.xlabel('uniqueness')
plt.colorbar()
plt.show()


plt.imshow(heatmap100H, cmap = 'Reds')
plt.xticks(range(4), xLabel)
plt.yticks(range(4), yLabel)
plt.ylabel('responsiveness')
plt.xlabel('uniqueness')
plt.colorbar()
plt.show()

'''
#-------------------------------------------------------------------------\
#entropy

heatmap90E = [[1100, 1000, 500, 400],
            [500, 200, 300, 600],
            [400, 700, 1400, 1300],
            [300, 1100, 1500, 1500]]



heatmap100E = [[3600, 3300, 1700, 500],
            [2000, 3100, 1500, 300],
            [5000, 5100, 1700, 300],
            [3600, 5300, 2200, 800]]

xLabel = [0.2, 0.4, 0.6, 0.8]
yLabel = [0.2, 0.4, 0.6, 0.8]

heatmap90E = numpy.array(heatmap90E)
heatmap100E = numpy.array(heatmap100E)

heatmap90E = heatmap90E/10000
heatmap100E = heatmap100E/10000

plt.imshow(heatmap90E, cmap = 'Reds')

plt.xticks(range(4), xLabel)
plt.yticks(range(4), yLabel)

plt.ylabel('responsiveness')
plt.xlabel('uniqueness')
plt.colorbar()
plt.show()


plt.imshow(heatmap100E, cmap = 'Reds')
plt.xticks(range(4), xLabel)
plt.yticks(range(4), yLabel)
plt.xlabel('responsiveness')
plt.ylabel('uniqueness')
plt.colorbar()
plt.show()

#-------------------------------------------------------------------------\
#entropy

heatmap90EH = heatmap90E - heatmap90H



heatmap100EH = heatmap100E - heatmap100H

xLabel = [0.2, 0.4, 0.6, 0.8]
yLabel = [0.2, 0.4, 0.6, 0.8]



plt.imshow(heatmap90EH, cmap = 'Reds')

plt.xticks(range(4), xLabel)
plt.yticks(range(4), yLabel)

plt.ylabel('responsiveness')
plt.xlabel('uniqueness')
plt.colorbar()
plt.show()


plt.imshow(heatmap100EH, cmap = 'Reds')
plt.xticks(range(4), xLabel)
plt.yticks(range(4), yLabel)
plt.xlabel('responsiveness')
plt.ylabel('uniqueness')
plt.colorbar()
plt.show()
'''