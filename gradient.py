import numpy as np
import math
import matplotlib.pyplot as plt
import time
import random
# import torch
# import torch.nn as nn

def rotate_points(points, vector):
    # Calculate the angle between the vector and the x-axis
    angle = np.arctan2(vector[1], vector[0])

    # Create the rotation matrix
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    rot_matrix = np.array([[cos_theta, -sin_theta],
                           [sin_theta, cos_theta]])

    # Perform the rotation
    rotated_points = np.dot(points, rot_matrix)
    return rotated_points

def gradient(x,y):
    filt = np.ones(29)/29
    y_smooth = np.convolve(y,filt,mode='valid')
    x = x[14:-14]
    dys = np.gradient(y_smooth,x)
    return x,dys

def gradient2(features,target,num):
    filt = np.ones(29)/29
    for i in range(num):
        target = np.append(np.array([0 for i in range(14)]),np.append(target,np.array([0 for i in range(14)])))
        target = np.convolve(target,filt,mode='valid')

    dy = np.array([np.nan_to_num(np.gradient(target,feature, edge_order=1),nan=float('inf')) for feature in features])
    return dy



# def descend(x,dy,n):
#     '''
#     n is the number of splits we want. Keep initializing random points and do descend until we get n splits 
#     '''
#     lr = 1
#     #initialise points by equally spacing them out
#     cxs = [int(0+(i*len(dy)/n)) for i in range(n)]
#     print('Initial points')
#     for cx in cxs:
#         print(x[cx])
#     for i in range(len(cxs)):
#         while(not(-0.8<dy[cxs[i]]<0.8)):
#             print(f'dy of {cxs[i]}: {dy[cxs[i]]}')
#             cxs[i] = int(cx + np.sign(dy[cxs[i]])*(-lr))
#             if cxs[i]<0 or cxs[i]>=len(dy):
#                 break
#             print(x[cxs[i]])
#     return cxs    


def descend(x,dy,n):
    
    #initialise points by equally spacing them out
    cxs = [int(0+(i*len(dy)/n)) for i in range(n)]
    for i,cx in enumerate(cxs):
        lr = 100
        j = 1
        while(not -0.5<dy[cx]<0.5):
            cx = int(cx + np.sign(dy[cx])*(-lr))
            if cx<0 or cx>=len(dy):
                break
            cxs[i] = cx
            print(cxs) 
            lr = math.ceil(100*(1.1**-j))+1
            j+=1
            print(lr) 
            time.sleep(0.1)
    return(cxs) 

def ascend(x,dy,n):
    lr = 100
    #initialise points by equally spacing them out
    cxs = [int(0+(i*len(dy)/n)) for i in range(n)]
    for cx in range(len(cxs)):
        while(not -0.3<dy[cxs[cx]]<0.3):
            print(dy[cxs[cx]])
            cxs[cx] = int(cxs[cx] + np.sign(dy[cxs[cx]])*(lr))
            if cxs[cx]<0 or cxs[cx]>=len(dy):
                break
            print(cxs)   
    return(cxs)     


def descend2(X,dy,n):
    lr = 1
    points = np.array([[int(0+(i*len(x)/n)) for i in range(1,n)] for x in X])

    print(points)

    for i in range(len(points)):
        for j in range(len(points[0])-1):
            while(not -0.8<dy[i][points[i][j]]<0.8):
                points[i][j] = int(points[i][j] + np.sign(dy[i][points[i][j]])*(-lr))
                print(points)   
    return points
    

def distance_from_line(slope_vector, point_on_line, point_list):
    distances = []
    
    # Extracting slope components
    slope_x, slope_y = slope_vector
    
    # Extracting point coordinates
    x0, y0 = point_on_line
    
    # Calculating the constant term (c) in the line equation: ax + by + c = 0
    c = -(slope_x * x0) - (slope_y * y0)
    
    for point in point_list:
        # Extracting point coordinates
        x, y = point
        
        # Calculating the distance between the point and the line
        distance = abs((slope_x * x) + (slope_y * y) + c) / math.sqrt((slope_x ** 2) + (slope_y ** 2))
        
        distances.append(distance)
    
    return np.array(distances)

def pad_and_derivate(dist):
    filt = np.ones(5)/5

    for i in range(5):
        y = np.append(np.array([0 for i in range(2)]),np.append(dist,np.array([0 for i in range(2)]),axis=0))
        y = np.convolve(y,filt,mode='valid')
    x = len(dist)   
    dy = np.gradient(y,x)

    ind = []

    for i in range(len(dy)-1):
        if dy[i]*dy[i+1] < 0:
            ind.append(i) if dist[i]>dist[i+1] else ind.append(i+1)

    return ind


# if __name__ == '__main__':
#     X_original = [
#     nn.Parameter(torch.linspace(-5, 5, 50, requires_grad=True)),
#     nn.Parameter(torch.linspace(-5, 5, 50, requires_grad=True))
# ]
#     X = []
#     X.append(nn.Parameter(torch.clone(X_original[0])))
#     X.append(nn.Parameter(torch.clone(X_original[1])))
#     Y = X[0]**2 + X[1]**2

#     learning_rate = 0.1
#     num_iterations = 1000

#     optimizer = torch.optim.SGD([X[0], X[1]], lr=learning_rate)

#     for _ in range(num_iterations):
#         optimizer.zero_grad()  # Clear the gradients

#         Y = X[0]**2 + X[1]**2  # Recalculate Y after any update to X[0] or X[1]

#         Y.backward(torch.ones_like(Y))  # Backpropagate to compute gradients

#         optimizer.step()  # Update the paramters

#     print(X[0])
#     print(X_original[0])
#     # Access the minimum coordinates
#     min_index_x0 = X[0].argmin().item()
#     min_index_x1 = X[1].argmin().item()

#     min_coord_x0 = X_original[0][min_index_x0].item()
#     min_coord_x1 = X_original[1][min_index_x1].item()

#     print("Minimum coordinate on X[0]:", min_coord_x0)
#     print("Minimum coordinate on X[1]:", min_coord_x1)

