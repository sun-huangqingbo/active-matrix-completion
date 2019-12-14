import matplotlib.pyplot as plt

urban = []
greenland = []
shrub = []
farm = []

with open(file="plotinfo.txt") as f:
    for line in f:
        ints = line.split(',')
        urban.append(ints[0])
        greenland.append(ints[1])
        shrub.append(ints[2])
        farm.append(ints[3])


plt.plot(range(len(urban)), urban, label = 'urban', c='#FF00FF')
plt.plot(range(len(greenland)), greenland, label = 'greenland', c='g')
plt.plot(range(len(shrub)), shrub, label = 'shrub', c='#00FFFF')
plt.plot(range(len(farm)), farm, label = 'farm', c='yellow')
plt.xlabel('Ggenerations')
plt.ylabel('Counts')
plt.title('Plot of Cell Counts')
plt.show()