import numpy as np
import matplotlib.pyplot as plt
import math
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR


def smoothen(y, window_size=29):
    """
    Smoothes an array of values using a moving average filter.

    Args:
        y (np.ndarray): Array of values to be smoothened.
        window_size (int, optional): Size of the moving average window. Defaults to 3.

    Returns:
        np.ndarray: Smoothened array of values.
    """

    # Check if the window size is valid
    if window_size < 2 or not isinstance(window_size, int):
        raise ValueError("Window size must be an integer greater than or equal to 2.")

    # Pad the array to handle boundary cases
    padding = window_size // 2
    y_padded = np.pad(y, padding, mode='edge')

    # Apply the moving average filter
    smoothed_values = np.convolve(y_padded, np.ones(window_size) / window_size, mode='valid')

    return smoothed_values


def pca(points):
	points = points / np.linalg.norm(points)
	cov = np.cov(points.T)
	eigenvalues, eigenvectors = np.linalg.eig(cov)
	idx = eigenvalues.argsort()[::-1]
	eigenvectors = eigenvectors[:,idx]
	return eigenvectors

def am(lst):
	return sum(lst)/len(lst)

dist = lambda a,b: math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

def plot_line_through_point(p,m,label):
	x = np.linspace(-20, 20, 10)
	y = m[1]/m[0] * (x - p[0]) + p[1]

	plt.arrow(0,0,m[0],m[1],color='orange')

	plt.plot(x, y, label=label)
	plt.plot(p[0], p[1], 'ro', label=f'AM({p[0]:.2f},{p[1]:.2f})')

  
def gradient2(features,target,num=1,n = 29):
    filt = np.ones(n)/n
    for i in range(num):
        target = np.append(np.array([0 for i in range(n//2)]),np.append(target,np.array([0 for i in range(n//2)])))
        target = np.convolve(target,filt,mode='valid')

    dy = np.array([np.nan_to_num(np.gradient(target,feature, edge_order=1),nan=0.0) for feature in features])
    return dy

def descend(points,n=15): #points m,3
  if points.shape[1]==1 or len(points.shape) == 1:
    return points
  points_t = points # points_t m,3 points 3,m

  points = points_t[np.argsort(points[:,0])].T
  grad = gradient2(points[0],points[-1])[0]

  cxs = [int(0+(i*len(grad)/n)) for i in range(n)]

  for i,cx in enumerate(cxs):
    lr = int(len(grad)*0.1)
    j=1
    while(not grad[cx]*grad[cx+1] < 0):
      cx = int(cx + np.sign(grad[cx])*(-lr))
      if cx<0 or cx>=len(grad):
        break
      cxs[i] = cx
      lr = math.ceil(100*(1.1**-j))+1
      j+=1

  filter_coords = [points[0][cx] for cx in cxs]

  # filt_points = np.zeros((1,points_t.shape[1]))
  filt_points = []

  # if condition soch na about shape for coordintes searching
  for coord in filter_coords:
    # filt_points = np.vstack((filt_points,points_t[points_t[:, 0] == coord]))
    filt_points.append(points_t[points_t[:,0]==coord])


  if points_t.shape[1] == 2:
    return np.array(filt_points).reshape((n,2))

  # filt_points = filt_points[1:]
  # print("Shape of points found ",filt_points.shape)
  final_points = []

  for group in filt_points:
    fixed_coord = group[0][0]
    sub_group = descend(group[:,1:])
    final_points.append(np.insert(sub_group,0,fixed_coord,axis=1))


  return np.array(final_points)

def ascend(points,n=15): #points m,3
  if points.shape[1]==1 or len(points.shape) == 1:
    return points
  points_t = points # points_t m,3 points 3,m

  points = points_t[np.argsort(points[:,0])].T
  grad = gradient2(points[0],points[-1])[0]

  cxs = [int(0+(i*len(grad)/n)) for i in range(n)]

  for i,cx in enumerate(cxs):
    lr = int(len(grad)*0.1)
    j=1
    while(not grad[cx]*grad[cx+1] < 0):
  
      cx = int(cx + np.sign(grad[cx])*(lr))
      if cx<0 or cx>=len(grad):
        break
      cxs[i] = cx
      lr = math.ceil(100*(1.1**-j))+1
      j+=1

  filter_coords = [points[0][cx] for cx in cxs]

  # filt_points = np.zeros((1,points_t.shape[1]))
  filt_points = []


  # if condition soch na about shape for coordintes searching
  for coord in filter_coords:
    # filt_points = np.vstack((filt_points,points_t[points_t[:, 0] == coord]))
    filt_points.append(points_t[points_t[:,0]==coord])


  if points_t.shape[1] == 2:
    return np.array(filt_points).reshape((n,2))

  # filt_points = filt_points[1:]
  # print("Shape of points found ",filt_points.shape)
  final_points = []

  for group in filt_points:
    fixed_coord = group[0][0]
    sub_group = descend(group[:,1:])
    final_points.append(np.insert(sub_group,0,fixed_coord,axis=1))


  return np.array(final_points)

def point_plane_side(point_to_test, normal, point_on_plane):
    """
    Determines which side of the plane the given point lies.

    Args:
        normal (list or numpy.ndarray): Normal vector of the plane.
        point_on_plane (list or numpy.ndarray): A point that lies on the plane.
        point_to_test (list or numpy.ndarray): The point to be tested.

    Returns:
        int: -1 if the point is on the negative side of the plane,
             1 if it is on the positive side, and 0 if it lies on the plane itself.
    """
    normal = np.array(normal)
    point_on_plane = np.array(point_on_plane)
    point_to_test = np.array(point_to_test)

    # Compute the signed distance from the point to the plane
    distance = np.dot(normal, point_to_test - point_on_plane)

    if np.isclose(distance, 0):
        return 0  # The point lies on the plane
    elif distance < 0:
        return -1  # The point lies on the negative side
    else:
        return 1  # The point lies on the positive side
    
def distance_from_hyperplane(point, normal_vector, point_on_plane):
    # Normalize the normal vector
    normalized_normal = normal_vector / np.linalg.norm(normal_vector)

    # Calculate the vector between the input point and the point on the plane
    vector_to_point = point - point_on_plane

    # Calculate the dot product of the vector and the normalized normal vector
    dot_product = np.dot(vector_to_point, normalized_normal)

    # Calculate the distance between the point and the plane
    distance = np.abs(dot_product)

    return distance   

def get_furthest_in_group(group,normal,point_on_plane):
  point_on_plane = np.array(point_on_plane)
  dist_func = partial(distance_from_hyperplane, normal_vector = normal,point_on_plane=point_on_plane)
  dist = list(map(dist_func,group))

  furthest = 0
  max_dist = dist[furthest]
  for i in range(1,len(group)):
     if dist[i] > max_dist:
        max_dist = dist[i]
        furthest = i
  return group[furthest]

def project_points(points, vector):
    # Ensure points and vector are NumPy arrays
    points = np.array(points)
    vector = np.array(vector)
    
    # Normalize the vector
    vector_norm = vector / np.linalg.norm(vector)
    
    # Calculate the dot product between each point and the vector
    dot_products = np.dot(points, vector_norm)
    
    # Calculate the projected points
    projected_points = dot_products[:, np.newaxis] * vector_norm
    
    return projected_points



def sort_points(points, vector):
    points = np.array(points)
    # Normalize the vector
    vector = vector / np.linalg.norm(vector)

    # Calculate the dot product of each point with the vector
    dot_products = np.dot(points, vector)

    # Sort the points based on the dot product values
    sorted_indices = np.argsort(dot_products)

    # Convert the sorted_indices to integer array
    sorted_indices = sorted_indices.astype(int)

    # Return the sorted points
    sorted_points = points[sorted_indices]
    return sorted_points

def check_concentration(vectors,points,normal,point_on_plane):
  avg_conc = 0
  vec_index = 0
  final_groups = []
  for x,vector in enumerate(vectors):
      sorted_points = sort_points(points,vector)
      side = point_plane_side(point_on_plane=point_on_plane,normal=normal,point_to_test=sorted_points[0])
      groups = [[sorted_points[0]]]
      j=0
      for i in range(1,len(sorted_points)):
        if side == point_plane_side(sorted_points[i],normal,point_on_plane):
           groups[j].append(sorted_points[i])
        else:
           j += 1
           side *= -1
           groups.append([sorted_points[i]])
        if i == len(sorted_points)-1:
           groups[j].append(sorted_points[i])

      group_lens = [len(group) for group in groups] 
      vec_conc = np.average(np.array(group_lens))
      if vec_conc > avg_conc:
         avg_conc = vec_conc
         vec_index = x
         final_groups = groups
  return vec_index,final_groups


def plot_plane(X,Y,ax,plane_normal,point_on_plane):
  pZ = (-plane_normal[0] * (X - point_on_plane[0]) - plane_normal[1] * (Y - point_on_plane[1])) / plane_normal[2] + point_on_plane[2]
  ax.plot_surface(X, Y, pZ, alpha=0.2)

def determine_region(point, planes, normal_vectors):
    n = len(planes)
    regions = []

    # Calculate the signed distances from the planes to the point
    for i,plane in enumerate(planes):
        signed_distance = np.dot(plane - point, normal_vectors[i])
        if signed_distance > 0:
            regions.append('Above')
        elif signed_distance < 0:
            regions.append('Below')
        else:
            regions.append('Above')

    # Determine the region the point lies in
    num_above = regions.count('Above')
    num_below = regions.count('Below')

    if num_above == 0:
        return 0

    if num_below == 0:
        return len(planes) 

    return num_above

def train_linear(data):
   X,y = data[:,:-1],data[:,-1]
   X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
   model = LinearRegression()
   model.fit(X_train,y_train)
   preds = model.predict(X_test)
   error = mean_absolute_error(y_test,preds)
   return model, error

def train_svr(dataset):
   X,y = dataset[:,:-1],dataset[:,-1]
   X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
   model = SVR()
   model.fit(X_train,y_train)
   preds = model.predict(X_test)
   error = mean_absolute_error(y_test,preds)
   return model, error
   