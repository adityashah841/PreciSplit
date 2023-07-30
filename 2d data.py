import matplotlib.pyplot as plt
import numpy as np
import random 
import math
from gradient import *
from utils import *

# Setting random seed
random.seed(42)

# initialising test points
r = np.linspace(0,14,100000)
x = np.array([i for i in r])
# y = -0.5x^3+0.1x^2+5x-2 with noise
# y = np.array([-0.5*((i + random.uniform(-1,1))**3)+0.1*((i + random.uniform(-1,1))**2) + 5*(i + random.uniform(-1,1))-2 for i in x]) 
# y = np.array([5*(np.sin(i)+random.uniform(-0.1,0.1)) - 0.5*(np.cos(2*i)+random.uniform(-0.1,0.1)) + i+random.uniform(-0.1,0.1) for i in x])
y = np.array([np.sin(3* i**0.5 + random.uniform(-0.1,0.1) )-np.cos(0.5*i + random.uniform(-0.1,0.1)) + 4 for i in x])
points = np.array(list(zip(x,y)))

# filtering points by removing outliers
f_points, outliers = filter(points)

# plotting filtered and outlier points
# plt.plot(f_points[:,0],f_points[:,1],'o--',color='g')
plt.scatter(outliers[:,0],outliers[:,1],color='b')
plt.plot([i for i in r],[np.sin(3* i**0.5)-np.cos(0.5*i) + 4 for i in r],label = 'original')

# calculating hyperplane via pca and plotting it through am of points
amx = am(f_points[:,0])
amy = am(f_points[:,1])
egv = pca(f_points)[0]
plot_line_through_point((amx,amy),egv,'hyperplane')



# calculate derivative

dy = gradient2([points[:,0]],points[:,1],3).squeeze()


# get splits
split = descend(points[:,0],dy,10)
# split_2 = ascend(points[:,0],dy,10)
split_points = np.array([[points[point,0], points[point,1]] for point in split])
# split_points2 = np.array([[points[point,0], points[point,1]] for point in split_2])
print("Moved points")
print(split_points)
# dist = distance_from_line(egv, (amx,amy), split_points)

# d_dist = pad_and_derivate(dist)
# print(d_dist)
# plt.scatter(split_points[dist.argmax()][0],split_points[dist.argmax()][1],label="split", color = 'purple')


# plot splits
# for i in d_dist:
# 	plt.scatter(split_points[i,0], split_points[i,1],color = 'green')

for point in split_points:
    plt.scatter(point[0],point[1],color='green')
    
# for point in split_points2:
#     plt.scatter(point[0],point[1],color='red')	

plt.axhline(0, color='black')
plt.axvline(0, color='black')

plt.show()

# Pass lines through maximas/minimas, 
# divide data using a plane passing through the line perpendicular to the hyperplane