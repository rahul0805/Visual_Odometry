#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
sys.path.remove(sys.path[1])

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

import open3d as o3d
from PIL import Image as im
from scipy.spatial.transform import Rotation as R
from scipy.sparse import lil_matrix
import time
from scipy.optimize import least_squares
from csv import writer

cy = 670/4
height = 65.5
imWidth = 3584/4
imHeight  = 1218/4
cx = imWidth / 12
f = (cx * 3**0.5)
print(f)
H_prev = np.eye(3)
txs = []
tys = []
thetas = []
stamps = []
nFeatures = []
timestamps = np.loadtxt("/home/rahul/dreamvu/visual_odo/dataset/indoor/op/timestamps.txt")
img1 =  None
mask1 = None

# clahe = cv2.createCLAHE(clipLimit=5.0)
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
orb = cv2.ORB_create(nfeatures = 100000)
# print(f)
# print(cx)
# print(cy)
# In[3]:




### Script for storing the pcd in the mesh file 
ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')




# In[4]:


def get_imgs(fname):
    print(fname)
    if(fname>1600):
        fileName1 = '/home/rahul/dreamvu/visual_odo/dataset/outdoor/depth_files/full3/fusion_highres_'+str(fname)+'.bin'
    else:
        fileName1 = '/home/rahul/dreamvu/visual_odo/dataset/outdoor/depth_files/full/fusion_highres_'+str(fname)+'.bin'
    depth1 = np.fromfile(fileName1, '<f4').reshape(1218,3584)

    depth1 = cv2.resize(depth1,(int(imWidth), int(imHeight)), interpolation = cv2.INTER_NEAREST)
    depth = depth1.ravel()
    my_ind = np.where((depth>=250))[0]
    depth[my_ind] = depth[my_ind]*0
    # for i in range(len(depth)):
    #     if(depth[i]>350):
    #         depth[i] = 0
    depth1 = depth.reshape( int(imHeight), int(imWidth))
    # depth1[0:int(imHeight/2),:] = depth1[0:int(imHeight/2),:]*0
    # print("in the input")
    my_ind = np.where(depth1>250)[0]
    # print("wrong detections are: ", len(my_ind))
    img1 = cv2.imread("/home/rahul/dreamvu/visual_odo/dataset/outdoor/imgs/{}.png".format(fname))
    mask1 = ((depth1>0)).astype(int)*250
    mask1 = cv2.resize(mask1,(int(imWidth), int(imHeight)))
    img1 = cv2.resize(img1, (int(imWidth), int(imHeight)),interpolation = cv2.INTER_NEAREST)
    mask1 = cv2.inRange(mask1, 150, 255, cv2.THRESH_BINARY)
    mask1 = cv2.dilate(mask1, np.ones((15,15)))
    return img1, mask1, depth1


# In[75]:

# In[19]:


## f2m implementation

def get_3d(points2D, depth1, f, cx, cy):
    num_pts = points2D.shape[0]
    us = points2D.reshape(-1,2)[:,0]
    vs = points2D.reshape(-1,2)[:,1]
    # print(np.max(depth1))
    Zs = depth1[vs.astype(int),us.astype(int)]
    # print(np.min(Zs))
    wrong = np.where(Zs[:]>=250)[0]
    Zs[wrong] = 0
    wrong = np.where(Zs[:]>=250)[0]
    # print("wrong detections" + str(wrong))
    thetas = (us / (2*cx)).astype(int) * np.pi / 3 
    us_ = us.copy()
    us = us % (2*cx)
    Xs = (us-cx)*Zs/f
    # print(np.max(Xs))
    vs_ = vs.copy()
    vs = vs % (2*cy)
    Ys = (vs-cy)*Zs/f
    new_Xs = Xs * np.cos(thetas) + Zs * np.sin(thetas) 
    new_Zs = -Xs * np.sin(thetas) + Zs * np.cos(thetas)
    points3D = np.zeros((num_pts,3), np.float32)
    points3D[:,0] = new_Xs.ravel()
    points3D[:,1] = Ys.ravel()
    points3D[:,2] = new_Zs.ravel()
    
    points3D = points3D.reshape(-1,3)
    return points3D/100, Zs.ravel()

def get_features(img1,mask1, depth1):
    colors = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1 = clahe.apply(img1)
    kp, des = orb.detectAndCompute(img1, mask=mask1)
    # print(des)
    #des = np.float32(des)
    # print(des)
    kp_pts = np.float32([ kp[m].pt for m in range(len(kp)) ]).reshape(-1,2)
    col = []
    for i in range(len(kp)):
        col.append(colors[kp_pts[i,1].astype(int), kp_pts[i,0].astype(int)])
    col = np.array(col)
    kp_2d = []
    for m in range(len(kp)):
        kp_2d.append([int(kp[m].pt[0]), int(kp[m].pt[1]), int(kp[m].pt[0]/(2*cx))])
    kp_2d = np.array(kp_2d).reshape(-1,3)
    kp_3d, my_z = get_3d(kp_2d[:,0:2], depth1, f, cx, cy)
    my_ind = np.where(kp_3d[:,2]!=0)[0]
    new_kp_3d = kp_3d[my_ind,:]
    new_kp_2d = kp_2d[my_ind,:]
    new_des = des[my_ind,:]
    new_col = col[my_ind,:]
    new_z = my_z[my_ind]
    uni_3d = np.unique(new_kp_3d, return_index= True, axis=0)[1]
    new_kp_3d1 = new_kp_3d[uni_3d,:]
    new_kp_2d1 = new_kp_2d[uni_3d,:]
    new_des1 = new_des[uni_3d,:]
    new_col1 = new_col[uni_3d,:]
    new_z1 = new_z[uni_3d] 
    # s = np.square(new_kp_3d1)
    # norm = np.sqrt(s[:,0]+s[:,1]+s[:,2])
    # less_ind = np.where(norm[:]<=3.5)[0]
    # new_kp_3d2 = new_kp_3d1[less_ind,:]
    # new_kp_2d2 = new_kp_2d1[less_ind,:]
    # new_des2 = new_des1[less_ind,:]
    # new_col2 = new_col1[less_ind,:]
    # new_z2 = new_z1[less_ind]
    # return kp_3d, kp_2d, des, col
    return new_kp_3d1, new_kp_2d1, new_des1, new_col1, new_z1
    # return new_kp_3d2, new_kp_2d2, new_des2, new_col2, new_z2

def match_features(kp1, kp2, des1, des2):
    
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                    table_number = 6, # 12
                    key_size = 12,     # 20
                    multi_probe_level = 2) #2
    search_params = dict(checks=50)   # or pass empty dictionary
  
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    # print(des1)
    # print(des2)
    # des1 = np.float32(des1)
    # des2 = np.float32(des2)
    matches = flann.knnMatch(des1,des2,k=2)
    good = []
    for (m,n) in matches:
        if m.distance < 0.9*n.distance:
            good.append(m)

    if  len(good) < 20:
        return []
    
    src_pts1 = np.float32([ kp1[m.queryIdx] for m in good ])
    dst_pts1 = np.float32([ kp2[m.trainIdx] for m in good ])
    return good

def get_tracking(good, kp_pts3d1, kp_pts3d2, my_z, kp_2, my_img, H_test):
    
    src_pts = np.float32([ kp_pts3d1[m.queryIdx] for m in good ])
    dst_pts = np.float32([ kp_pts3d2[m.trainIdx] for m in good ])
    my_z2 = np.float32([ my_z[m.trainIdx] for m in good ])
    theta2 = np.float32([ kp_2[m.trainIdx,2] for m in good ])
    
    count = 0
    src = []
    dst = []
    src_1 = []
    dst_1 = []
    sc = []
    for i in range(len(src_pts)):
        src_1.append([src_pts[i,0], src_pts[i,2]])
        dst_1.append([dst_pts[i,0], dst_pts[i,2]])
            
        if((abs(src_pts[i,1]-dst_pts[i,1])<0.1) and (src_pts[i,2]!=0) and (dst_pts[i,2]!= 0) ):
            count+=1
            src.append([src_pts[i,0], src_pts[i,2]])
            dst.append([dst_pts[i,0], dst_pts[i,2]])
            sc.append(1)
        else:
            sc.append(0)
        
    src = np.asarray(src).reshape(-1,2)
    dst = np.asarray(dst).reshape(-1,2)
    src_1 = np.asarray(src_1).reshape(-1,2)
    dst_1 = np.asarray(dst_1).reshape(-1,2)
    print("2D transform is : ")
    H1,score = cv2.estimateAffinePartial2D(src, dst, ransacReprojThreshold=0.150)
    print(H1)
    theta = np.arctan2(H1[0,1], H1[0,0])
    scale = H1[0,0] / np.cos(theta)
    if(yes==1):
        in_score.append(np.sum(score))
        in_len.append(len(score))
        in_ratio.append(np.sum(score)/len(score))
    tx = H1[0,2]
    ty = H1[1,2]
    score1 = score.copy()
    new_H = np.eye(3);
    new_H[0,0] = np.cos(theta)
    new_H[0,1] = np.sin(theta)
    new_H[0,2] = tx
    new_H[1,0] = -np.sin(theta)
    new_H[1,1] = np.cos(theta)
    new_H[1,2] = ty
    src1 = (new_H[0:2,0:2]@src_1.T).T 
    t = np.array([tx, ty])
    dst1 = src1+t
    diff = (dst_1-dst1)**2
    s = np.sqrt((diff[:,0]+diff[:,1])/2)
    score = (s<0.15).astype(int)
    score1 = score.copy()
    err1 = s

    ##Calculating 3D error for GT
    R1 = np.eye(2)
    R1[0,0]= H_test[0,0]
    R1[0,1] = H_test[0,2]
    R1[1,0] = H_test[2,0]
    R1[1,1] = H_test[2,2]
    t = np.array([H_test[0,3], H_test[2,3]])
    src1 = (R1@src_1.T).T 
    dst1 = src1+t
    diff = (dst_1-dst1)**2
    s = np.sqrt((diff[:,0]+diff[:,1]))
    score = (s<0.15).astype(int)
    err2 = s
    
    #2D reprojection for VO
    HH = np.eye(3)
    HH[0,0] = H1[0,0]
    HH[2,2] = H1[1,1]
    HH[0,2] = H1[0,1]
    HH[2,0] = H1[1,0]
    r = R.from_matrix(HH)
    camera_params = np.array([[r.as_rotvec()[0], r.as_rotvec()[1], r.as_rotvec()[2],H1[0,2],0,H1[1,2],f,0,0]])
    sr_2d = project(src_pts,camera_params , theta2)
    ds_2d = project(dst_pts, np.array([[0,0.0001,0,0.0001,0,0,f,0,0]]), theta2)
    diff = (sr_2d-ds_2d)**2
    s = np.sqrt((diff[:,0]+diff[:,1]))
    
    err = 0
    c  = 0
    pp = []
    marker = np.zeros((len(ds_2d)))
    img3 = my_img.copy()
    for ite in range(len(ds_2d)):
        x = int(ds_2d[ite,0]+2*cx*theta2[ite]+cx)
        y = int(ds_2d[ite,1] + cy)
        x1 = int(sr_2d[ite,0]+2*cx*theta2[ite]+cx)
        y1 = int(sr_2d[ite,1] + cy)
        start = (x,y)
        end = (x1,y1)
        if(np.sqrt((x1-x)**2+(y1-y)**2)<10):
            err += np.sqrt((x1-x)**2+(y1-y)**2)
            img3 = cv2.line(img3,start,end,(255,0,0), 1)
            img3 = cv2.circle(img3,(x,y),2,(0,255,255),-1)
            img3 = cv2.circle(img3,(x1,y1),2,(0,0,255),-1)
            c+=1
            pp.append(ite)
            marker[ite]+=20
    print("2D points are",c)
    print("#d points lenght are:, ", len(err1[pp]))
    print("Total No of Matched Features: ",len(ds_2d) )
    print("REprojection error is: ", np.mean(err) )
    print("3D REprojection error is: ", np.mean(err1[pp]) )
    H1,_ = cv2.estimateAffinePartial2D(src_1[pp], dst_1[pp], ransacReprojThreshold=0.30)
    
    theta = np.arctan2(H1[0,1], H1[0,0])
    scale = H1[0,0] / np.cos(theta)
    if(yes==1):
        in_score.append(np.sum(score))
        in_len.append(len(score))
        in_ratio.append(np.sum(score)/len(score))
    tx = H1[0,2]
    ty = H1[1,2]
    H_after = np.eye(3);
    H_after[0,0] = np.cos(theta)
    H_after[0,1] = np.sin(theta)
    H_after[0,2] = tx
    H_after[1,0] = -np.sin(theta)
    H_after[1,1] = np.cos(theta)
    H_after[1,2] = ty
    
    op1 = c
    op2 = err/c
    op3 = np.mean(err1[pp])   

    r = R.from_matrix(H_test[0:3,0:3])
    camera_params = np.array([[r.as_rotvec()[0], r.as_rotvec()[1], r.as_rotvec()[2],H_test[0,3],0, H_test[2,3],f,0,0]])
    sr_2d = project(src_pts,camera_params , theta2)
    ds_2d = project(dst_pts, np.array([[0,0.0001,0,0.0001,0,0,f,0,0]]), theta2)
    diff = (sr_2d-ds_2d)**2
    s = np.sqrt((diff[:,0]+diff[:,1]))
    print(np.max(s))
    print(np.median(s))
    print(np.mean(s))
    pp =np.where(score>0)
    # print(pp)
    print("SEE HERE")
    print(np.mean(s[pp]))
    print(np.mean(s))
    err = 0
    c  = 0
    img3 = my_img.copy()
    pp = []
    for ite in range(len(ds_2d)):
        x = int(ds_2d[ite,0]+2*cx*theta2[ite]+cx)
        y = int(ds_2d[ite,1] + cy)
        # img3 = cv2.circle(img3,(x,y),2,(0,255,255),-1)
        x1 = int(sr_2d[ite,0]+2*cx*theta2[ite]+cx)
        y1 = int(sr_2d[ite,1] + cy)
        start = (x,y)
        end = (x1,y1)
        if(np.sqrt((x1-x)**2+(y1-y)**2)<10):
            err += np.sqrt((x1-x)**2+(y1-y)**2)
            img3 = cv2.line(img3,start,end,(255,0,0), 1)
            img3 = cv2.circle(img3,(x,y),2,(0,255,255),-1)
            img3 = cv2.circle(img3,(x1,y1),2,(0,0,255),-1)
            c+=1
            pp.append(ite)
            marker[ite]+=10
    H1,_ = cv2.estimateAffinePartial2D(src_1[pp], dst_1[pp], ransacReprojThreshold=0.30)
    theta = np.arctan2(H1[0,1], H1[0,0])
    scale = H1[0,0] / np.cos(theta)
    tx = H1[0,2]
    ty = H1[1,2]
    H_after_gt = np.eye(3);
    H_after_gt[0,0] = np.cos(theta)
    H_after_gt[0,1] = np.sin(theta)
    H_after_gt[0,2] = tx
    H_after_gt[1,0] = -np.sin(theta)
    H_after_gt[1,1] = np.cos(theta)
    H_after_gt[1,2] = ty
    
    print("No of points are: ", c)
    print("REprojection error is: ", err/c )
    print("3D REprojection error is: ", np.mean(err2[pp]) )
    # filename = './reproj_GT.jpg'
    # cv2.imwrite(filename, img3)

    img3 = my_img.copy()
    # pp = []
    op8 = 0
    op9 = 0
    op10 = 0
    for ite in range(len(ds_2d)):
        x = int(ds_2d[ite,0]+2*cx*theta2[ite]+cx)
        y = int(ds_2d[ite,1] + cy)
        if(marker[ite]==10):
            img3 = cv2.circle(img3,(x,y),2,(0,0,255),-1)
            op8+=1
        if(marker[ite]==20):
            img3 = cv2.circle(img3,(x,y),2,(0,255,255),-1)
            op9+=1
        if(marker[ite]==30):
            img3 = cv2.circle(img3,(x,y),2,(0,255,0),-1)
            op10 +=1
    filename = './match_checker/'+str(my_count)+'.jpg'
    cv2.imwrite(filename, img3)
    op4 = c
    op5 = err/c
    op6 = np.mean(err1[pp])
    print(op8)
    print(op9)
    print(op10)
    return H_after_gt, score1,H_after_gt, len(ds_2d), op1, op2, op3, op4, op5, op6, op8, op9, op10
    # return H_after, score1

def get_metrics(H):
    theta = np.arctan2(H[0,1], H[0,0])
    scale = H[0,0] / np.cos(theta)
    tx = H[0,2]
    ty = H[1,2]	
    return tx,ty,theta
def get_3d_H(H1):
    H_fin = [H1[0,0], 0, H1[0,1], H1[0,2], 0, 1, 0, 0, H1[1,0], 0, H1[1,1], H1[1,2]]
    H_fin = np.array(H_fin).reshape(3,4)
    return H_fin


# ## Bundle Adjustment inbuilt


# In[8]:


def rotate1(points, rot_vecs):
    for i in range(len(rot_vecs)):
        H = rot_vecs[i].reshape(3,4)
        s = ((H[0:3,0:3]@(points[i].T))+H[:,3]).T
        points[i] = s
    return points

def rotate(points, rot_vecs):
   
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis] #np.newaxis converts this into a column vector.
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
        
    check = (theta!=0).astype(int)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return (cos_theta * points) + check*(((1 - cos_theta) * v * dot) + (sin_theta * np.cross(v, points)))

def project(points, camera_params, theta):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj1 = rotate(points[:,0:3], camera_params[:, :3])
    points_proj1 += camera_params[:, 3:6]
    thetas = theta * np.pi / 3 
    points_proj = np.copy(points_proj1)
    points_proj[:,0] = points_proj1[:,0]*np.cos(thetas) - points_proj1[:,2]*np.sin(thetas)
    points_proj[:,2] = points_proj1[:,0]*np.sin(thetas) + points_proj1[:,2]*np.cos(thetas)
    for i in range(len(points_proj)):
        if(points_proj[i,2]==0):
            points_proj[i,0] = 0
            points_proj[i,1] = 0
            points_proj[i,2] = 1
    points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj**2, axis=1) #n = (r_c)^2
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj



# In[9]:


def project1(points, camera_params, theta):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj1 = rotate1(points[:,0:3], camera_params)
    thetas = theta * np.pi / 3 
    points_proj = np.copy(points_proj1)
    points_proj[:,0] = points_proj1[:,0]*np.cos(thetas) - points_proj1[:,2]*np.sin(thetas)
    points_proj[:,2] = points_proj1[:,0]*np.sin(thetas) + points_proj1[:,2]*np.cos(thetas)
    for i in range(len(points_proj)):
        if(points_proj[i,2]==0):
            points_proj[i,0] = 0
            points_proj[i,1] = 0
            points_proj[i,2] = 1
    points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    k1 = 0
    k2 = 0
    n = np.sum(points_proj**2, axis=1) #n = (r_c)^2
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj


# In[10]:


def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d, theta):
    """Compute residuals.
    
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices], theta)
    return (points_proj - points_2d).ravel()


# In[11]:


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2 
    
    n = n_cameras * 9 + n_points * 3 
    
    A = lil_matrix((m, n), dtype=float)

    i = np.arange(camera_indices.size)
    for s in [1,3,5]:
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1

    for s in [0,2]:
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1
        
    return A


# In[12]:


def fun1(params, n_cameras, n_points, camera_indices, point_indices, points_2d, theta):
    """Compute residuals.
    
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 12].reshape((n_cameras, 12))
    points_3d = params[n_cameras * 12:].reshape((n_points, 3))
    points_proj = project1(points_3d[point_indices], camera_params[camera_indices], theta[point_indices])
    return (points_proj - points_2d).ravel()


# In[13]:


def bundle_adjustment_sparsity1(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2 
    
    n = n_cameras * 12 + n_points * 3 
    
    A = lil_matrix((m, n), dtype=float)

    i = np.arange(camera_indices.size)
    for s in [0,2,3,8,10,11]:
        A[2 * i, camera_indices * 12 + s] = 0.1
        A[2 * i + 1, camera_indices * 12 + s] = 0.1

    for s in range(3):
        A[2 * i, n_cameras * 12 + point_indices * 3 + s] = 0.01
        A[2 * i + 1, n_cameras * 12 + point_indices * 3 + s] = 0.01
        
    return A


# In[14]:




def get_things1(kp_3d, kp_2d, des, comp_list, H, map_3d, map_2d, map_des, map_cam, map_view, my_max):
    points_3d = []
    points_2d = []
    camera_ind = []
    points_ind = []
    cam_params = []
    
    dst_3d = kp_3d
    dst_2d = kp_2d
    src_3d = map_3d
    src_2d = map_2d
    src_cam = map_cam
    low_bound = []
    up_bound = []
    my_min = 0
    for i in range(my_min,my_max+1):
        cam_param = [map_view[i,0], map_view[i,1], map_view[i,2], map_view[i,3], map_view[i,4], map_view[i,5], f,0,0]
        cam_params.append(cam_param)

        low_bound.append(-np.pi)
        low_bound.append(-np.pi)
        low_bound.append(-np.pi)
        low_bound.append(-20)
        low_bound.append(-np.inf)
        low_bound.append(-20)
        low_bound.append(f-1)
        low_bound.append(-1)
        low_bound.append(-1)
        up_bound.append(np.pi)
        up_bound.append(np.pi)
        up_bound.append(np.pi)
        up_bound.append(20)
        up_bound.append(np.inf)
        up_bound.append(20)
        up_bound.append(f)
        up_bound.append(0)
        up_bound.append(0)
    r = (R.from_matrix((H[0:3, 0:3]))).as_rotvec()
    t = H[:,3]
    # cam_param = [H[0,0], H[0,1], H[0,2], H[0,3], H[1,0], H[1,1], H[1,2], H[1,3], H[2,0], H[2,1], H[2,2], H[2,3]]
    cam_param = [r[0], r[1], r[2], t[0], t[1], t[2], f, 0, 0]
    cam_params.append(cam_param)
    
    low_bound.append(-np.pi)
    low_bound.append(-np.pi)
    low_bound.append(-np.pi)
    low_bound.append(-20)
    low_bound.append(-np.inf)
    low_bound.append(-20)
    low_bound.append(f-1)
    low_bound.append(-1)
    low_bound.append(-1)
    up_bound.append(np.pi)
    up_bound.append(np.pi)
    up_bound.append(np.pi)
    up_bound.append(20)
    up_bound.append(np.inf)
    up_bound.append(20)
    up_bound.append(f)
    up_bound.append(0)
    up_bound.append(0)
    new_cam = len(cam_params)-1
    cam_params = np.array(cam_params).reshape(-1,9)
    count = 0
    l1 = []
    l2 = []
    count = 0
    for m in comp_list:
        count+=1
        l1.append(m.queryIdx)
        l2.append(m.trainIdx)
    l1 = np.array(l1).reshape(1,-1)
    l2 = np.array(l2).reshape(1,-1)
    l = np.vstack((l1,l2))
    l_fin = l[:,l[1, :].argsort()]
    j = 0
    count = len(points_3d)
    prev = -1
    final_l1 = []
    final_l2 = []
    final_des = []
    while(j<(len(l_fin[0]))):
        i1 = l_fin[0,j]
        i2 = l_fin[1,j]
        if(i2!=prev):
            check = 0
            for ii in range(len(src_2d[i1])):
                m_2d = src_2d[i1][ii]
                check = 1
                ind = int(src_cam[i1][ii])
                points_2d.append([int((m_2d[0]%(2*cx))-cx), int((m_2d[1]%(2*cy))-cy), m_2d[2]])
                points_ind.append(count)
                camera_ind.append(ind)
                final_l1.append(i1)
                final_l2.append(0)
            # x = ((map_des[i1]*len(src_2d[i1]))+des[i2])/(len(src_2d[i1])+1)
            # map_des[i1] = x
            if(check==1):
                points_2d.append([int((dst_2d[i2,0]%(2*cx))-cx), int((dst_2d[i2,1]%(2*cy))-cy),dst_2d[i2,2]])
                points_ind.append(count)
                camera_ind.append(new_cam)
                final_l1.append(i2)
                final_l2.append(1)
                wld_pt = src_3d[i1]
                points_3d.append([wld_pt[0], wld_pt[1], wld_pt[2]])
                prev = i2
                count = len(points_3d)
                low_bound.append(-20)
                low_bound.append(-np.inf)
                low_bound.append(-20)

                up_bound.append(20)
                up_bound.append(np.inf)
                up_bound.append(20)
        j+=1
        
    cam_params = np.array(cam_params).reshape(-1,9)
    points_3d = np.array(points_3d)
    # print(points_3d.shape)
    # print(np.unique(points_3d, axis=0).shape)
    points_2d = np.array(points_2d)
    camera_ind = np.array(camera_ind).reshape(len(camera_ind))
    points_ind = np.array(points_ind).reshape(len(points_ind))
    final_l1 = np.array(final_l1)
    final_l2 = np.array(final_l2)
    return cam_params, points_3d, points_2d, camera_ind, points_ind, final_l1, final_l2, low_bound, up_bound, map_des


def do_BA2(kp_3d, kp_2d, des, comp_list, H, map_3d, map_2d, map_des, map_cam, map_view, my_update, col, col2, my_max, BA=0):
    #print("yes")
    # print("Collecting Input")
    # print("Length of map before BA", len(map_3d))
    camera_params, points_3d, points_2d, camera_ind, points_ind, final_l1, final_l2, low_bound, up_bound, map_des = get_things1(kp_3d, kp_2d, des, comp_list, H, map_3d, map_2d, map_des, map_cam, map_view, my_max)
    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]
    n = 9 * n_cameras + 3 * n_points
    m = 2 * points_2d.shape[0]
    # print("Starting Bundle Adjustment")
    # print("n_cameras: {}".format(n_cameras))
    # print("n_points: {}".format(n_points))
    # print("Total number of parameters: {}".format(n))
    # print("Total number of residuals: {}".format(m))
    
    x0 = np.hstack((camera_params.ravel(), points_3d[:, 0:3].ravel()))
    resx = x0.copy()
    if(BA==1):
        # f0 = fun(x0, n_cameras, n_points, camera_ind, points_ind, points_2d, points_3d[:,3])
        f0 = fun(x0, n_cameras, n_points, camera_ind, points_ind, points_2d[:,:2], points_2d[:,2])
        A = bundle_adjustment_sparsity(n_cameras, n_points, camera_ind, points_ind)
        t0 = time.time()

        res = least_squares(fun, x0, jac_sparsity=A, bounds=(low_bound, up_bound), verbose=1, x_scale='jac', ftol=1e-4, method='trf',
                            args=(n_cameras, n_points, camera_ind, points_ind, points_2d[:,:2], points_2d[:,2]))
        t1 = time.time()

        resx = res.x

    # print("Optimization done")
    my_min = 0
    my_max = np.max(camera_ind)+1
    H_op = np.zeros((3,4))
    #H_op = np.array(resx[(my_max-1)*9:(my_max-1)*9+6])
    H_op[0:3,0:3] = R.from_rotvec(resx[(my_max-1)*9:(my_max-1)*9+3]).as_matrix()
    H_op[0:3,3] = resx[(my_max-1)*9+3:(my_max-1)*9+6]
    # print(H_op)
    final_pts = np.array(resx[my_max*9:]).reshape(-1,3)
    ini_pts = np.array(x0[my_max*9:]).reshape(-1,3)
    map_view = np.vstack((map_view,resx[(my_max-1)*9:(my_max-1)*9+6]))
    for i in range(my_min,my_max-1):
        # if()
        map_view[i] = resx[i*9 : i*9+6]
        # print(map_view[i])
    update_list = []
    for i in range(len(final_l1)):
        if(final_l2[i]==1):
            update_list.append(final_l1[i])
        if(final_l2[i]==0):
            # camera_params = camera_params[:n_cameras * 9].reshape((n_cameras, 9))
            # p_2d = project(final_pts[points_ind[i]], camera_params[camera_ind[i]], points_2d[i,2])
            # p_2d1 = project(ini_pts[points_ind[i]], camera_params[camera_ind[i]], points_2d[i,2])
            # e1 = np.sum(np.square((p_2d-points_2d[i,:2]).ravel()))
            # e2 = np.sum(np.square((p_2d1-points_2d[i,:2]).ravel()))
            err = np.sqrt(np.sum(np.square((final_pts[points_ind[i]] - ini_pts[points_ind[i]]).ravel()))/3)
            if(err<0.5):
                # print("updated")
                # print(final_pts[points_ind[i]])
                map_3d[final_l1[i]] = final_pts[points_ind[i]]
            #print(points_ind[i])
            # map_des[final_l1[i]] = 
            map_cam[final_l1[i]].append(my_max-1)
    
    update_list = np.array(update_list)
    print("Points Updated")
    # print("Length of map after BA", len(map_3d))
    if(my_update==1):
        l1 = []
        l2 = []
        new_3d = []
        new_2d = []
        new_cam = []
        new_view = []
        new_des = []
        new_col = []
        l2 = np.unique(np.sort(update_list))
        j = 0
        for i in range(len(kp_2d)):
            if(i == l2[j]):
                j += 1
                if(j==len(l2)):
                    j = 0
            else:
                pt = (np.linalg.inv(H_op[0:3,0:3])@(kp_3d[i].T - H_op[:,3]))
                new_3d.append(pt)
                new_2d = []
                new_cam = []
                new_des.append(des[i])
                new_2d.append(kp_2d[i])
                new_cam.append(my_max-1)
                new_col.append(col2[i])
                map_2d.append(new_2d)
                map_cam.append(new_cam)
        
        new_3d = np.array(new_3d)
        new_des = np.array(new_des)
        print(new_des)
        print(len(kp_2d))
        print(len(l2))
        # new_2d = np.array(new_2d)
        # new_cam = np.array(new_cam)
        new_col = np.array(new_col)
        # print("Length of common pts: ", len(l2))
        # print("Length of new pts: ", len(new_3d))
        # print("Length of new image: ", len(kp_3d))
        # print(new_3d.shape)
        map_3d = np.vstack((map_3d,new_3d))
        # map_2d = np.vstack((map_2d,new_2d))
        map_des = np.vstack((map_des,new_des))
        # map_cam = np.hstack((map_cam,new_cam))
        col = np.vstack((col,new_col))
        # print("Length of map after update", len(map_3d))
        # print("New Points Added")
    return H_op, map_3d, map_2d, map_des, map_cam, map_view, col, my_max-1, len(l2) 
 




# In[23]:


def plot_img(good1,img2,kp_2d2,num, mask2, sc):
    k = np.array([kp_2d2[m.trainIdx,0:2] for m in good1])
    img3 = img2.copy()
    for ii in range(len(k)):
        x = int(k[ii,0])
        y = int(k[ii,1])
        img3 = cv2.circle(img2,(x,y),2,(0,255,255),-1)
        if((sc[ii])):
            # print("saving")
            # print(np.sum(sc[ii]))
            img3 = cv2.circle(img2,(x,y),2,(0,0, 255),-1)
    # print(sc)
    filename = './match/'+str(num)+'.jpg'
    cv2.imwrite(filename, img3)  


# In[24]:


def reproj(map_3d, good1, H):
    map_3d_com = np.array([map_3d[m.queryIdx] for m in good1])
    map_2d_com = np.array([map_2d[m.queryIdx] for m in good1])
    proj_pts = project1(map_3d_com, H, map_2d_com[:,2])
    for i in range(len(proj_pts)):
        proj_pts[i,1] = proj[i,1]+cy
        proj_pts[i,0] = (proj[i,0]+cx) + 2*cx*map_2d_com[i,2]
    
    
def num_match(comp_list,kp_2d, map_2d):
    l1 = []
    l2 = []
    for m in comp_list:
        if(kp_2d[m.trainIdx,2] == map_2d[m.queryIdx,2]):
            l1.append(m.queryIdx)
            l2.append(m.trainIdx)
    
    l1 = np.array(l1)
    l2 = np.array(l2)
    l2 = np.unique(np.sort(l2))
    return len(l2)
def write_data(row):
    with open('event.csv', 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(row)
        f_object.close()
    return
def write_data1(row):
    with open('data_match.csv', 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(row)
        f_object.close()
    return
# In[25]:


init = 0

H_ini = []
H_list = []
second_point = 0 
col = []

txs = []
tys = []
thetas = []
match_nums = []
not_match_nums = []
map_fea = []
# give the address of the images in the get_imgs functions
i =0
H_prev = np.array([1,0,0,0,1,0,0,0,1]).reshape(3,3)
x_prev = 0
y_prev = 0
in_score = []
in_len = []
in_ratio = []
my_count = 10
odometry = np.fromfile(str("turtle_odom.bin"), dtype=np.float32).reshape(3,-1)
odometry = odometry.T
tx_gt = []
ty_gt = []
while(i<2491):
    if (init==0):
        yes = 0
        first_point = i
        print("image" + str(i) + " is processing")
        # Map initialisation
        #Image 1
        second_point = i+10 ## Second frame for map initialisation
        img1, mask1, depth1 = get_imgs(i)
        kp_3d1,kp_2d1, des1, col1, my_z = get_features(img1,mask1,depth1)
        H_fin = np.array([1,0,0,0,0,1,0,0,0,0,1,0]).reshape(3,4)
        map_view = []
        # map_view.append([1,0,0,0,0,1,0,0,0,0,1,0])
        # map_view = np.array(map_view).reshape(-1,12)
        map_view.append([0,0,0,0,0,0])
        map_view = np.array(map_view).reshape(-1,6)
        map_3d = np.array(kp_3d1)
        map_cam = []
        des_list = []
        # map_2d = np.array(kp_2d1)
        map_2d = []
        for j in range(len(kp_3d1)):
            l1 = []
            l2 = []
            d =[]
            d.append(des1[j])
            l1.append(kp_2d1[j])
            l2.append(0)
            des_list.append(d)
            map_2d.append(l1)
            map_cam.append(l2)
        map_des = np.array(des1)
        # print(len(map_cam[0]))
        # print(map_cam[0][0])
        # print(map_cam[4][1])
        # map_des = des1
        H = np.eye(4)
        H[0,0] = np.cos(odometry[i,2])
        H[0,2] = -np.sin(odometry[i,2])
        H[2,2] = np.cos(odometry[i,2])
        H[2,0] = np.sin(odometry[i,2])
        H[0,3] = -odometry[i,1]
        H[2,3] = odometry[i,0]
        H_add = H
        
        col = np.array(col1)
        H_list.append(H_fin)
        H_ini.append(H_fin)
        init = 1
        # break
        # Image 2
        img2, mask2, depth2 = get_imgs(second_point)
        kp_3d2, kp_2d2, des2, col2, my_z = get_features(img2,mask2,depth2)
        good1 = match_features(map_2d, kp_2d2, map_des, des2)
        H = np.eye(4)
        H[0,0] = np.cos(odometry[second_point,2])
        H[0,2] = -np.sin(odometry[second_point,2])
        H[2,2] = np.cos(odometry[second_point,2])
        H[2,0] = np.sin(odometry[second_point,2])
        H[0,3] = -odometry[second_point,1]
        H[2,3] = odometry[second_point,0]
        H1 = np.linalg.inv(H)@H_add
        H_ori = H1[0:3,0:4]
        
        H_2d, sc, _, op7, op1, op2, op3, op4, op5, op6, op8, op9, op10 = get_tracking(good1, map_3d, kp_3d2, my_z, kp_2d2, img2, H_ori)
        H_fin = get_3d_H(H_2d)
        print(len(map_des))
        
        print("length of the Map: ", len(map_3d))
        #Bundle adjustment
        H_op, map_3d, map_2d, map_des, map_cam, map_view, col, my_max, my_match = do_BA2(kp_3d2, kp_2d2, des2, good1, H_fin, map_3d, map_2d, map_des, map_cam, map_view, 1, col, col2,0,0)
        #result = do_BA(kp_3d2, kp_2d2, des2, good1, H_fin, map_3d, map_2d, map_des, map_cam, map_view, my_update=1)
        col = np.array(col)
        final_pts = map_3d
        # print(np.unique(map_cam))
        # print("length of the Map: ", len(map_3d))
        # print("length of the new pts: ", len(kp_3d2))
        print(len(des2))
        print(len(map_des))

        final_col = col
        file_name = './init_pcd/map_init'+str(i)+'.ply'
        write_ply(file_name,final_pts,final_col)
        print("Load a ply point cloud, print it, and render it")
        
        # i+=1
        # break
        
    elif(i%10==0):
        if(i!=1830):
            yes = 1
            print("image" + str(i) + " is processing")
            img2, mask2, depth2 = get_imgs(i)
            kp_3d2, kp_2d2, des2, col2, my_z = get_features(img2,mask2,depth2) # getting ORB features from the img
            print(len(des2))
            print(len(map_des))
            good1 = match_features(map_2d, kp_2d2, map_des, des2) # matches features and output the matches
            # get_tracking(good, kp_pts3d1, kp_pts3d2, H_test, my_z, kp_1, kp_2, my_img, col1, col2, prev_img)

            H = np.eye(4)
            H[0,0] = np.cos(odometry[i,2])
            H[0,2] = -np.sin(odometry[i,2])
            H[2,2] = np.cos(odometry[i,2])
            H[2,0] = np.sin(odometry[i,2])
            H[0,3] = -odometry[i,1]
            H[2,3] = odometry[i,0]
            H1 = np.linalg.inv(H)@H_add
            H_ori = H1[0:3,0:4]
            H_2d, sc, _, op7, op1, op2, op3, op4, op5, op6, op8, op9, op10 = get_tracking(good1, map_3d, kp_3d2, my_z, kp_2d2, img2, H_ori) # f2f tracking returns 2D transformation
            plot_img(good1,img2,kp_2d2,i, mask2, sc)
            H_fin = get_3d_H(H_2d)
            print("This is important")
            print(H_fin)
            print(H_ori)
            r = R.from_rotvec([0, np.pi/4, 0])
            H_extra = r.as_matrix()
            # H_fin = H_extra@H_ori
            # H_fin = H_extra@H_ori
            H_ini.append(H_fin)
            my_count += 10
            # print("length of the Map: ", len(map_3d))
            # print("length of the new pts: ", len(kp_3d2))
            # print("length of the common pts: ", len(good1)) 
            
            if(((i%10)==0) and (i!=second_point) and (i!=first_point)):
                print(H_fin)
                # The below function call makes bundle adjustment call and updates map with new values
                # if map update is not required the make the variable my_update = 0
                # H_op, map_3d, map_2d, map_des, map_cam, map_view, col = without_BA(kp_3d2, kp_2d2, des2, good1, H_fin, map_3d, map_2d, map_des, map_cam, map_view, 1, col, col2)
                # H_op, map_3d, map_2d, map_des, map_cam, map_view, col, my_max = without_BA(kp_3d2, kp_2d2, des2, good1, H_fin, map_3d, map_2d, map_des, map_cam, map_view, 1, col, col2)
                H_op, map_3d, map_2d, map_des, map_cam, map_view, col, my_max, my_match = do_BA2(kp_3d2, kp_2d2, des2, good1, H_fin, map_3d, map_2d, map_des, map_cam, map_view, 1, col, col2,my_max,0)
                H_fin = H_op
                # print(np.unique(map_cam))
                
                # print("Updated length of the Map: ", len(map_3d))
                col = np.array(col)
                final_pts = map_3d

                final_col = col
            if(i%30==0):
                file_name = './pcd/map_0_'+str(i)+'_10.ply'
                # write_ply(file_name,final_pts,final_col)
                # print("Load a ply point cloud, print it, and render it")
                # print("The length of map is: ")
            # print(map_3d.shape)
            t_bot = H_extra@H[0:3,3]
            H_list.append(H_fin)
            H_fin1 = np.linalg.inv(H_fin[0:3,0:3]@H_prev)@(H_fin[:,3])
            H_fin2 = H_fin[0:3,0:3]@H_prev
            theta = np.arctan2(-H_fin2[0,2], H_fin2[0,0])
            tx = -H_fin1[0]
            ty = -H_fin1[2]
            # if(i%40==0):
            #     print(tx,ty)
            #     break
            txs.append(tx+x_prev)
            tys.append(ty+y_prev)
            tx_gt.append(t_bot[0])
            ty_gt.append(t_bot[2])
            thetas.append(theta)
            #my_match = num_match(good1,kp_2d2, map_2d)
            #match_nums.append(my_match)
            # row = [i,my_match,tx,ty,theta]
            # write_data(row)
            row = [i,op7, op8, op9, op10,tx+x_prev,ty+y_prev,theta]
            write_data1(row)
            map_fea.append(len(map_des))
            plt.axis("equal")
            plt.scatter(txs, tys)
            plt.scatter(tx_gt, ty_gt)
            plt.savefig("./traj/traj_"+str(i)+".png")
            plt.savefig("./traj/traj.png")
            plt.close("all")
            txs1 = np.array(txs)
            tys1 = np.array(tys)
            thetas1 = np.array(thetas)
            match_nums1 = np.array(match_nums)
            not_match_nums1 = np.array(not_match_nums)
            map_fea1 = np.array(map_fea)
            np.save('x_co1', txs1)
            np.save('y_co1', tys1)
            np.save('theta_co1', thetas1)
            np.save('matchs_1', match_nums1)
            np.save('not_matchs_1', not_match_nums1)
            np.save('total_fea_1', map_fea1)
            np.save('in_score', in_score)
            np.save('in_len', in_len)
            np.save('in_ratio', in_ratio)
        if((i%30==0) and (i!=first_point) and (i!=2490)):
            print("yes")
            init = 0
            H_prev = H_fin[0:3,0:3]@H_prev
            x_prev = tx+x_prev
            y_prev = ty+y_prev
            i -= 10
    i +=10

txs = np.array(txs)
tys = np.array(tys)
thetas = np.array(thetas)
match_nums = np.array(match_nums)
not_match_nums = np.array(not_match_nums)
map_fea = np.array(map_fea)
np.save('theta_co', thetas)
np.save('matches', match_nums)
np.save('x_co', txs)
np.save('y_co', tys)
# for xx in range(len(map_2d)):
#     this_row = map_2d[xx]
#     this_cam = map_cam[xx]
#     # print(this_cam)
#     # print(xx)
#     for yy in range(len(this_row)):
#         # print(this_cam[yy])
#         # print(this_row[yy][0])
#         saver[0,xx,this_cam[yy]] = this_row[yy][0]
#         saver[1,xx,this_cam[yy]] = this_row[yy][1]
#         saver[2,xx,this_cam[yy]] = 255
# np.save('./coros/'+str(i)+'_map.npy',saver)

