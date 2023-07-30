import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils import *
import time

np.random.seed(42)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Create a grid of x and y values
x = np.linspace(-40, 40, 200)
y = np.linspace(-40, 40, 200)
X, Y = np.meshgrid(x, y)

# Define the z coordinate as a combination of hills and valleys
Z = 0.3*X-0.5*(abs(y)**0.5) + np.cos(0.5*X)**2 + 3*np.sin(0.5*Y)

points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
start = time.time()
pca = pca(points)
egv = pca[-1]
mean = (am(points[:,0]),am(points[:,1]),am(points[:,2]))

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.scatter(X, Y, Z,' o',color = 'blue',alpha=0.05)
# ax.scatter(mean[0],mean[1],mean[2],' o',color='black')

min_points = np.unique(descend(points).reshape((-1,3)),axis=0)
max_points = np.unique(ascend(points).reshape((-1,3)),axis=0)
extrema = np.vstack((min_points,max_points))

positive_points = []
negative_points = []

for point in extrema:
    side = point_plane_side(point,egv,mean)
    if side ==1 :
        positive_points.append(point)
    else:
        negative_points.append(point)  


# Create and fit the KMeans model
kmeans_pos = KMeans(n_clusters=len(positive_points)//5)
kmeans_pos.fit(positive_points)
kmeans_neg = KMeans(n_clusters=len(negative_points)//5)
kmeans_neg.fit(negative_points)

# Retrieve cluster labels
cluster_labels_pos = kmeans_pos.labels_  
cluster_labels_neg = kmeans_neg.labels_  

groups_pos = [[] for i in range(len(positive_points)//5)]
groups_neg = [[] for i in range(len(negative_points)//5)]

for i,point in enumerate(positive_points):
    groups_pos[cluster_labels_pos[i]].append(point)
for i,point in enumerate(negative_points):
    groups_neg[cluster_labels_neg[i]].append(point)    

groups_pos.extend(groups_neg)

final_points = []

for group in groups_pos:
    final_points.append(get_furthest_in_group(group,egv,mean))

index, groups = check_concentration(pca[:-1],final_points,egv,mean)

plane_points = []
normals = []
vector = pca[index]
for group in groups:
    if len(group)>1:
        plane_points.append(group[0])
        normals.append(np.cross(group[0]-group[1],egv))


# final_points = sort_points(final_points,vector)    
# ax.scatter(final_points[:,0],final_points[:,1],final_points[:,2],c=colors, cmap='viridis')
# ax.quiver(mean[0], mean[1], mean[2], vector[0], vector[1], vector[2], color='red', label='Arrow')
ax.quiver(mean[0], mean[1], mean[2], egv[0], egv[1], egv[2], color='red', label='Arrow')


color_list = ["red", "yellow", "green", "blue", "indigo", "violet", "pink", "purple", "brown", "black", "white", "gray"]

regions = []

for point in points:
    regions.append(determine_region(point,plane_points,normals))
     
normal = pca[index]
regions = np.array(regions)
points_regions = np.insert(points,points.shape[1],regions,axis=1)

dataset = [points[np.where(points_regions[:, -1] == i)[0]] for i in np.unique(regions)]
models = []

error = 0
for data in dataset:
    print(len(data))
    model, err = train_linear(data)
    models.append(model)
    error += err
print("Linear models avg error: ", error/len(dataset))
print("Time for linear models: ",time.time()-start)
start = time.time()
model, error = train_svr(points)
print("SVR error: ",error)
print("Time for SVR model: ",time.time()-start)


# for i,point in enumerate(points):
#     ax.scatter(point[0],point[1],point[2],c=color_list[regions[i]],alpha=0.1)

# plane_normal = egv
# point_on_plane = mean

# pZ = (-plane_normal[0] * (X - point_on_plane[0]) - plane_normal[1] * (Y - point_on_plane[1])) / plane_normal[2] + point_on_plane[2]
# ax.plot_surface(X, Y, pZ, alpha=0.4, color = 'blue')

# plt.show()