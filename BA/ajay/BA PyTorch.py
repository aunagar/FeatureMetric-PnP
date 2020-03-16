#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function


# In[2]:


import urllib
import bz2
import os
import numpy as np
import torch
from torch import autograd

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[3]:


device = torch.device('cpu')


# In[4]:


BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/ladybug/"
FILE_NAME = "problem-49-7776-pre.txt.bz2"
URL = BASE_URL + FILE_NAME


# In[5]:


if not os.path.isfile(FILE_NAME):
    urllib.request.urlretrieve(URL, FILE_NAME)


# In[6]:


def read_bal_data(file_name):
    with bz2.open(file_name, "rt") as file:
        n_cameras, n_points, n_observations = map(int, file.readline().split())
    
        camera_indices = np.empty(n_observations, dtype = int)
        point_indices = np.empty(n_observations, dtype = int)
        points_2d = torch.empty(n_observations, 2, device = device)

        for i in range(n_observations):
            camera_index, point_index, x, y = file.readline().split()
            camera_indices[i] = camera_index
            point_indices[i] = point_index
            points_2d[i] = torch.tensor([float(x), float(y)])

            camera_params = torch.empty(n_cameras*9, device = device)

        for i in range(n_cameras*9):
            camera_params[i] = float(file.readline())

        camera_params = camera_params.view(n_cameras, -1)

        points_3d = torch.empty(n_points*3, device = device)

        for i in range(n_points*3):
            points_3d[i] = float(file.readline())
        points_3d = points_3d.view(n_points, -1)
            
    return camera_params, points_3d, camera_indices, point_indices, points_2d


# In[7]:


c_params, p3d, camera_indices, point_indices, points_2d = read_bal_data(FILE_NAME)


# In[8]:


camera_indices = torch.tensor(camera_indices, device = device)
point_indices = torch.tensor(point_indices, device = device)


# In[9]:


n_cameras = c_params.size()[0]
n_points = p3d.size()[0]

n = 9*n_cameras + 3*n_points
m = 2*points_2d.size()[0]

print("n_cameras: {}".format(n_cameras))
print("n_points: {}".format(n_points))
print("Total number of parameters: {}".format(n))
print("Total number of residuals: {}".format(m))


# In[10]:


c_params.requires_grad_(True)
p3d.requires_grad_(True)


# In[11]:


def rotate(points, rot_vecs):
    
    theta = torch.norm(rot_vecs, dim = 1, keepdim=True)
    v = rot_vecs/theta
    v[v != v] = 0.
#     print(v.size(), points.size())
    
    dot = torch.sum(points*v, dim = 1, keepdim = True)
    
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    
    ans = cos_theta*points + sin_theta*torch.cross(v, points) + dot*(1-cos_theta)*v
    
    return ans


# In[12]:


def project(points, camera_params):
    R = torch.index_select(camera_params, 1, torch.tensor([0,1,2]))
    T = torch.index_select(camera_params, 1, torch.tensor([3,4,5]))
    # print(R.size(), T.size())
    points_proj = rotate(points, R)
    points_proj = points_proj + T
    denom = torch.index_select(points_proj,1,torch.tensor([2])).view(-1,1)
    points_proj_2 = -torch.index_select(points_proj,1,torch.tensor([0,1]))/denom
#     f = camera_params[:,6]
#     k1 = camera_params[:,7]
#     k2 = camera_params[:,8]
    f = torch.index_select(camera_params, 1, torch.tensor([6]))
    k1 = torch.index_select(camera_params, 1, torch.tensor([7]))
    k2 = torch.index_select(camera_params, 1, torch.tensor([8]))
    
    n = torch.sum(torch.mul(points_proj_2,points_proj_2), dim = 1)
    # print(f.size(), k1.size(), k2.size(), n.size())
    r = 1 + torch.mul(n,k1.view(-1)) + torch.mul(k2.view(-1),torch.mul(n,n))
    # print(r.size(), f.size(), points_proj.size())
    # print(torch.mul(r,f.view(f.numel())).size())
    points_proj_3 = points_proj_2*torch.mul(r,f.view(-1)).unsqueeze(1)
    return points_proj_3


# In[13]:


def fun(camera_params, points_3d, n_cameras, n_points, camera_indices, point_indices, points_2d):
#     cp = params[:n_cameras*9].view(n_cameras, 9)
#     p3d = params[n_cameras*9:].view(n_points, 3)
#     points_proj = project(p3d[point_indices], cp[camera_indices])
    points_3d_2 = torch.index_select(points_3d, 0, point_indices)
    camera_params_2 = torch.index_select(camera_params, 0, camera_indices)
    points_proj = project(points_3d_2, camera_params_2)
    ans = points_proj - points_2d
    return ans.view(-1)


# In[46]:


f0 = fun(camera_params, points_3d, n_cameras, n_points, camera_indices, point_indices, points_2d)


# In[47]:


f = fun(camera_params, points_3d, n_cameras, n_points, camera_indices, point_indices, points_2d)


# In[36]:


loss = f.pow(2).sum()


# In[54]:


points_3d = torch.index_select(p3d, 0, point_indices)
camera_params = torch.index_select(c_params, 0, camera_indices)


# In[55]:


R = torch.index_select(camera_params, 1, torch.tensor([0,1,2]))
T = torch.index_select(camera_params, 1, torch.tensor([3,4,5]))
theta = torch.norm(R, dim = 1, keepdim=True)
v = R/theta
dot = torch.sum(points_3d*v, dim = 1, keepdim = True)

cos_theta = torch.cos(theta)
sin_theta = torch.sin(theta)

points_proj = cos_theta*points_3d + sin_theta*torch.cross(v, points_3d) + dot*(1-cos_theta)*v
points_proj = points_proj + T
denom = torch.index_select(points_proj,1,torch.tensor([2])).view(-1,1)
points_proj = -torch.index_select(points_proj,1,torch.tensor([0,1]))/denom

f = torch.index_select(camera_params, 1, torch.tensor([6]))
k1 = torch.index_select(camera_params, 1, torch.tensor([7]))
k2 = torch.index_select(camera_params, 1, torch.tensor([8]))

n = torch.sum(torch.mul(points_proj,points_proj), dim = 1)
r = 1 + torch.mul(n,k1.view(-1)) + torch.mul(k2.view(-1),torch.mul(n,n))
points_proj = points_proj*torch.mul(r,f.view(-1)).unsqueeze(1)


# In[56]:


f = (points_proj - points_2d).view(-1)


# In[57]:


loss = f.pow(2).sum()


# In[58]:


loss.backward()


# In[61]:


print(p3d.grad)


# In[14]:


lr=1e-6


# In[16]:


for i in range(1):
    f = fun(c_params, p3d, n_cameras, n_points, camera_indices, point_indices, points_2d)
    
    loss = f.pow(2).sum()
    
    print(i, " --> ", loss.item())
    loss.backward()
    
    
    
    with torch.no_grad():
        print(c_params)
        c_params -= lr*c_params.grad
        p3d -= lr*p3d.grad
        print(c_params)
        c_params.grad.zero_()
        p3d.grad.zero_()


# In[60]:


current_f = loss.grad_fn
print(current_f)
while True:
    current_f = current_f.next_functions[0][0]
    print(current_f)


# In[ ]:




