import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils import *
import time

# np.random.seed(42)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Create a grid of x and y values
x = np.linspace(-10, 10, 40)
y = np.linspace(-10, 10, 40)
X, Y = np.meshgrid(x, y)

# Define the z coordinate as a combination of hills and valleys
Z = 0.3*X-0.5*(abs(y)**0.5) + np.cos(0.5*X)**2 + 3*np.sin(0.5*Y)

points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
start = time.time()
Pca = pca(points)
egv = Pca[-1]
mean = (am(points[:,0]),am(points[:,1]),am(points[:,2]))

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.scatter(X, Y, Z,' o',alpha=0.1)
ax.scatter(mean[0],mean[1],mean[2],' o',color='black')

# ax.scatter(points[:,0],points[:,1],points[:,2],alpha=0.1,c='b')

min_points = np.unique(descend(points).reshape((-1,3)),axis=0)
max_points = np.unique(ascend(points).reshape((-1,3)),axis=0)

# for point in np.vstack((min_points,max_points)):
#     ax.scatter(point[0],point[1],point[2],c='r')

positive_points = []
negative_points = []

for point in max_points:
    if point_plane_side(point,egv,mean) == 1:
      positive_points.append(point)
for point in min_points:
    if point_plane_side(point,egv,mean) == -1:
      negative_points.append(point)  


positive_points = np.array(positive_points)
negative_points = np.array(negative_points)

# ax.scatter(positive_points[:,0],positive_points[:,1],positive_points[:,2],c='g')
# ax.scatter(negative_points[:,0],negative_points[:,1],negative_points[:,2],c='r')

# Create and fit the KMeans model
kmeans_pos = KMeans(n_clusters=len(positive_points)//5)
kmeans_pos.fit(positive_points)
kmeans_neg = KMeans(n_clusters=len(negative_points)//5)
kmeans_neg.fit(negative_points)

# Retrieve cluster labels
cluster_labels_pos = kmeans_pos.labels_  
cluster_labels_neg = kmeans_neg.labels_  

# Convert the lists to NumPy arrays for better performance
positive_points = np.array(positive_points)
negative_points = np.array(negative_points)
cluster_labels_pos = np.array(cluster_labels_pos)
cluster_labels_neg = np.array(cluster_labels_neg)

# Calculate the number of clusters
num_clusters_pos = len(positive_points) // 5
num_clusters_neg = len(negative_points) // 5

# Initialize the groups
groups_pos = [np.empty((0, positive_points.shape[1])) for _ in range(num_clusters_pos)]
groups_neg = [np.empty((0, negative_points.shape[1])) for _ in range(num_clusters_neg)]

# Use advanced indexing to assign points to the respective clusters
for i in range(num_clusters_pos):
    mask = cluster_labels_pos == i
    groups_pos[i] = positive_points[mask]

for i in range(num_clusters_neg):
    mask = cluster_labels_neg == i
    groups_neg[i] = negative_points[mask]   

groups = groups_pos+groups_neg
del groups_neg
del groups_pos
print(len(groups))
final_points = np.zeros((len(groups), points.shape[1]))

colors = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # teal
    "#aec7e8",  # light blue
    "#ffbb78",  # light orange
    "#98df8a",  # light green
    "#ff9896",  # light red
    "#c5b0d5",  # light purple
]
for i,group in enumerate(groups):
    # ax.scatter(group[:,0],group[:,1],group[:,2],c=color_list[i])
    final_points[i] = get_furthest_in_group(group,egv,mean)
    color = np.random.rand(1,3)
    ax.scatter(final_points[i][0],final_points[i][1],final_points[i][2],color=color)

index, groups = check_concentration(Pca[:-1],final_points,egv,mean)

plane_points = []
normals = []
vector = Pca[index]
for group in groups:
    if len(group)>1:
        plane_points.append(group[0])
        normals.append(np.cross(group[0]-group[1],egv))
regions = []
for point in points:
    regions.append(determine_region(np.array(point),np.array(plane_points),np.array(normals)))
     
regions = np.array(regions)
points_regions = np.insert(points,points.shape[1],regions,axis=1)

dataset = [points[np.where(points_regions[:, -1] == i)[0]] for i in np.unique(regions)]
# for i,data in enumerate(dataset):
#     ax.scatter(data[:,0],data[:,1],data[:,2],c=color_list[i],alpha=0.5)
models = []
# print("Time for splitting: ",time.time()-start)
# error = 0
# for data in dataset:
#     print(len(data))
#     model, err = train_linear(data)
#     models.append(model)
#     error += err
# print("Linear models avg error: ", error/len(dataset))
# print("Time for linear models: ",time.time()-start)

plane_normal = egv
point_on_plane = mean

pZ = (-plane_normal[0] * (X - point_on_plane[0]) - plane_normal[1] * (Y - point_on_plane[1])) / plane_normal[2] + point_on_plane[2]
ax.plot_surface(X, Y, pZ, alpha=0.4, color = 'blue')
plt.show()