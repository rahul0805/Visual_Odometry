#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
sys.path.remove(sys.path[1])
import os
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

f = 525.0
cx = 319.5  # optical center x
cy = 239.5  # optical center y
    
print(f)
H_prev = np.eye(3)
txs = []
tys = []
thetas = []
stamps = []
nFeatures = []
img1 =  None
mask1 = None

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
orb = cv2.ORB_create(nfeatures = 10000)
left_dir = './rgb/' 
img_name = sorted(os.listdir(left_dir))
right_dir = './depth/'
depth_name = sorted(os.listdir(right_dir))
# print(img_name)
j = 0
i = 0
new_d = []
new_im = []

# Syncing the depth and images
while (i <(len(depth_name))):
    d = float(depth_name[i][:-4])
    im = float(img_name[j][:-4])
    diff= abs(im-d)
    im1 = float(img_name[j+1][:-4])
    diff1= abs(im1-d)
    if((diff>=diff1) and (im1<=d)):
        j +=1
    else:
        if(im<d):
            new_d.append(depth_name[i])
            new_im.append(img_name[j])
            j+=1
        i+=1


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



def get_imgs(fname):
    """
    Function takes input of as image_number and outputs the image, mask, depth. 
    Input:
        fname: image number
    Output:
        img1: image
        mask1: mask
        depth1: depth map
    """
    depth1 = cv2.imread(right_dir + new_d[fname], -1)
    img1 = cv2.imread(left_dir + new_im[fname])
    mask1 = ((depth1>0)).astype(int)*255
    mask1 = cv2.inRange(mask1, 150, 255, cv2.THRESH_BINARY)
    return img1, mask1, depth1


## f2m implementation

def convert_3d(points_2d, depth_image, image):
    """
    Function takes input of as image, depth, 2d_points and outputs the 3d_points. 
    Input:
        imgage: image
        points_2d: 2D points
        depth_image: depth map in size of image
    Output:
        points: 3d_points for points
        points_3d: 3d_points for all pixel points
        col: Colors for key point pixels
    """
    fx = 525.0  # focal length x
    fy = 525.0  # focal length y
    cx = 319.5  # optical center x
    cy = 239.5  # optical center y
    factor = 5000 # for the 16-bit PNG files
    points_3d = []
    cols = []
    colors = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for v in range(depth_image.shape[0]):
        for u in range(depth_image.shape[1]):
            Z = depth_image[v,u] / factor
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            points_3d.append([X,Y,Z])
            cols.append(colors[v,u])
    points = []
    for i in range(len(points_2d)):
        x = int(points_2d[i,0])
        y = int(points_2d[i,1])
        # print(y)
        Z = depth_image[y,x] / factor
        X = (x - cx) * Z / fx
        Y = (y - cy) * Z / fy
        points.append([X,Y,Z])
    points_3d = np.array(points_3d)
    cols = np.array(cols)
    points = np.array(points)
    
    return points, points_3d, cols

def get_features(img1,mask1, depth1):
    """
    Function takes input of as image, depth, mask and outputs the 3d_points, 2d_points, descriptors, colors. 
    Input:
        img1: image
        mask1: mask in size of image
        depth1: depth map in size of image
    Output:
        kp_3d: 3d_points for key points
        kp_2d: 2d_points for keypoints
        des: Descriptors for keypoints
        col: Colors for key point pixels
    """
    colors = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img3 = img1.copy()
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1 = clahe.apply(img1)    # Applying Clahe
    kp, des = orb.detectAndCompute(img1, mask=mask1)    # Computing ORB features
    kp_pts = np.float32([ kp[m].pt for m in range(len(kp))]).reshape(-1,2)
    # Getting Colors
    col = []
    for i in range(len(kp)):
        col.append(colors[kp_pts[i,1].astype(int), kp_pts[i,0].astype(int)])
    col = np.array(col)
    # Getting 2D points
    kp_2d = []
    for m in range(len(kp)):
        kp_2d.append([int(kp[m].pt[0]), int(kp[m].pt[1])])
    kp_2d = np.array(kp_2d).reshape(-1,2)
    
    # Getting the 3D points
    kp_3d, _, _ = convert_3d(kp_2d, depth1, img3)
    
    # Removing points with Zero depth
    my_ind = np.where(kp_3d[:,2]!=0)[0]
    new_kp_3d = kp_3d[my_ind,:]
    new_kp_2d = kp_2d[my_ind,:]
    new_des = des[my_ind,:]
    new_col = col[my_ind,:]
    
    # Removing the duplicates
    uni_3d = np.unique(new_kp_3d, return_index= True, axis=0)[1]
    new_kp_3d1 = new_kp_3d[uni_3d,:]
    new_kp_2d1 = new_kp_2d[uni_3d,:]
    new_des1 = new_des[uni_3d,:]
    new_col1 = new_col[uni_3d,:]
    return kp_3d, kp_2d, des, col

def match_features(kp1, kp2, des1, des2):
    """
    Function takes input of set of key_points and descriptors and outputs the matches array. 
    Input:
        kp1: Set of keypoints1
        kp2: set of keypoints2
        des1: set of descriptors1
        des2: Set of descriptors2
    Output:
        good: Array of common matches
    """
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                    table_number = 6, # 12
                    key_size = 12,     # 20
                    multi_probe_level = 2) #2

    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    good = []
    for (m,n) in matches:
        if m.distance < 0.8*n.distance: ## Lowe's ratio imp for tuning
            good.append(m)

    if  len(good) < 20:
        return []
    return good


def get_tracking(good, kp_pts3d1, kp_pts3d2, theta):
    """
    Function takes input of comparision_array, src_3d_pts, dst_3d_pts,theta(if calculated previously) and outputs the transformation, inliers array, scale, tx,ty, theta. 
    Input:
        H: 2D transformation  
    Output:
        new_H: 2D transformation from RANSAC
        score1: array to determine the inliers
        scale: scale
        tx: translation in x
        ty: translation in y
        theta: rotated angle

    """
    src_pts = np.float32([ kp_pts3d1[m.queryIdx] for m in good ])
    dst_pts = np.float32([ kp_pts3d2[m.trainIdx] for m in good ])
    
    count = 0
    src = []
    dst = []
    src_1 = []
    dst_1 = []
    sc = []
    for i in range(len(src_pts)):
        src_1.append([src_pts[i,0], src_pts[i,2]])
        dst_1.append([dst_pts[i,0], dst_pts[i,2]])
        if((abs(src_pts[i,1]-dst_pts[i,1])<0.05) and (src_pts[i,2]!=0) and (dst_pts[i,2]!= 0) ):
            count+=1
            src.append([src_pts[i,0], src_pts[i,2]])
            d = [dst_pts[i,0], dst_pts[i,2]]
            dst.append([d[0], d[1]])
            sc.append(1)
        else:
            sc.append(0)
    src = np.asarray(src).reshape(-1,2)
    dst = np.asarray(dst).reshape(-1,2)
    
    src_1 = np.asarray(src_1).reshape(-1,2)
    dst_1 = np.asarray(dst_1).reshape(-1,2)
    H1,score = cv2.estimateAffinePartial2D(src, dst, ransacReprojThreshold=0.50)
    
    theta = np.arctan2(H1[0,1], H1[0,0])
    scale = H1[0,0] / np.cos(theta)
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
    
    return new_H, score1, scale, tx, ty, theta

def get_metrics(H):
    """
    Function takes input of 2d_transformations and outputs the tx, ty, theta. 
    Input:
        H: 2D transformation  
    Output:
        tx: translation in x
        ty: translation in y
        theta: rotated angle
    """
    theta = np.arctan2(H[0,1], H[0,0])
    scale = H[0,0] / np.cos(theta)
    tx = H[0,2]
    ty = H[1,2]	
    return tx,ty,theta

def get_3d_H(H1):
    """
    Function takes input of 2d_transformations, and changes to 3D transformations. 
    Input:
        H1: 2D transformation  
    Output:
        H_fin: 3D transformation
    """
    H_fin = [H1[0,0], 0, H1[0,1], H1[0,2], 0, 1, 0, 0, H1[1,0], 0, H1[1,1], H1[1,2]]
    H_fin = np.array(H_fin).reshape(3,4)
    return H_fin


# Bundle Adjustment inbuilt

def rotate(points, rot_vecs):
    """
    Function takes input of 3d_points, transformations and rotate into ground frame 3-D points. 
    Input:
        points: 3D points in world frame
        rot_vecs : rotation vectors for th points  
    Output:
        rotated_points
    """
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
    """
    Function takes input of 3d_points, transformations and Convert 3-D points to 2-D by projecting onto images. 
    Input:
        points: 3D points in world frame
        camera_params: parameters of camera corrosponding to the point
        theta: Needed For PAL camera to specify the sub camera index for the points
    Output:
        points_proj: 2D reprojected points for 3D points 

    """
    # Convert the 3D points to Camera Frame by rotaion followes by translation
    points_proj1 = rotate(points[:,0:3], camera_params[:, :3])
    points_proj1 += camera_params[:, 3:6]
    # FOR PAL: Converting into the Sub-camera Frame by respective rotation
    thetas = theta * np.pi / 3 
    points_proj = np.copy(points_proj1)
    points_proj[:,0] = points_proj1[:,0]*np.cos(thetas) - points_proj1[:,2]*np.sin(thetas)
    points_proj[:,2] = points_proj1[:,0]*np.sin(thetas) + points_proj1[:,2]*np.cos(thetas)
    # Avoiding Zero error
    for i in range(len(points_proj)):
        if(points_proj[i,2]==0):
            points_proj[i,0] = 0
            points_proj[i,1] = 0
            points_proj[i,2] = 1
    # 2D projection
    points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj



def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d, theta):
    """
    Function takes input of cno of cameras, points_seen and their indices and returns the residual. 
    Input:
        params: contains camera parameters and 3d points; desired variable
        n_camera: integer representing no.of.cameras
        n_points: integer representing no.of.points
        camera_indices: array representing the camera index of 2d_points
        point_indices: array representing the point index for 2d_points
        Points_2d: 2d points
        theta: Needed For PAL camera to specify the sub camera index
    Output: Resisual, an array of size of length of points_2d

    """
    
    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices], theta)
    print("Residual is: ", (points_proj - points_2d).ravel())
    return (points_proj - points_2d).ravel()


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    """
    Function takes input of cno of cameras, points_seen and their indices and returns the Sparse Jacobian for BA. 
    Input:
        n_camera: integer representing no.of.cameras
        n_points: integer representing no.of.points
        camera_indices: array representing the camera index of 2d_points
        point_indices: array representing the point index for 2d_points
    Output:
        A: Jacobian of size (camera_indices.size * 2 x n_cameras * 9 + n_points * 3)
         
    """

    m = camera_indices.size * 2 
    n = n_cameras * 9 + n_points * 3 
    A = lil_matrix((m, n), dtype=float)

    i = np.arange(camera_indices.size)
    
    # Jacobian for Parameters update
    for s in [1,3,5]:
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1
    
    # Jacobian for 3D Points update
    for s in [0,2]:
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1
        
    return A


def get_things1(kp_3d, kp_2d, des, comp_list, H, map_3d, map_2d, map_des, map_cam, map_view, my_max):
    """
    Function takes input of current image features, estimated_transformation, map. 
    After processing outputs the 3d points, updated_2d_map points, camera_parameters in the format needed for Bundle adjustment Module 
    Input:
        kp_3d: 3d_points of current image (nx3)
        kp_2d: 2d_points of current image (nx3)
        des: descriptors (nx32)
        comp_list: Matches array from Flann based matcher 
        map_3d: 3d_points in map (mx3)
        map_2d: Corrosponding 2d_points for map_3d points (m rows and contains set of 3 sized array in each row depend on observations)
        map_des: map_desciptors for 3d_points (mx32)
        map_cam: camera in which map_3d points are viewed (m rows and contains set of 1 sized array in each row depend on observations)
        map_view: contains transformation of the cameras seens (mx6)
        my_max : (no of current camera - 1)
    
    cam_params, points_3d, points_2d, camera_ind, points_ind, final_l1, final_l2, low_bound, up_bound, map_des, src_2d
    
    Output:
        cam_params : camera transformations (no.of.cameras x 9)
        points_3d: 3d_points in matches (matchesx3)
        points_2d : Corrosponding 2d_points for 3d points 
        cam_ind: index of camera for 2d points 
        points_ind : index of 3d_points for 2d points
        final_l1: indexing for update
        final_l2: indexing for update (to seperate map and frame points)
        low_bound: Lower bounds for the updating variable
        up_bound: upper bounds for the updating variable
        map_des: Map descriptors
        src_2d: map_2d_points_updated
         
    """
    # Initializing the arrays
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

    # Updating the Camera parameters in map and setting the bounds for the update 
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
    
    # Updating the Camera parameters for frame and setting the bounds for the update
    r = (R.from_matrix((H[0:3, 0:3]))).as_rotvec()
    t = H[:,3]
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
    
    # listing variables to iterate 
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

    # Iterating through the list made and making sure no duplicates
    while(j<(len(l_fin[0]))):
        i1 = l_fin[0,j]
        i2 = l_fin[1,j]
        if(i2!=prev):
            # Map points insertion
            
            check = 0
            for ii in range(len(src_2d[i1])):
                m_2d = src_2d[i1][ii]
                check = 1
                ind = int(src_cam[i1][ii])
                points_2d.append([int((m_2d[0]%(2*cx))-cx), int((m_2d[1]%(2*cy))-cy),0])

                points_ind.append(count)
                camera_ind.append(ind)
            final_l1.append(i1)
            final_l2.append(0)
            
            # Taking Mean Desciptor if needed un comment 2 lines below
            # x = ((map_des[i1]*len(src_2d[i1]))+des[i2])/(len(src_2d[i1])+1)
            # map_des[i1] = x
            
            if(check==1):
                # Frame points insersion
                points_2d.append([int((dst_2d[i2,0])-cx), int((dst_2d[i2,1])-cy), 0])
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
                src_2d[i1].append([int((dst_2d[i2,0])), int((dst_2d[i2,1]))])
        j+=1
    
    # Final Output
    cam_params = np.array(cam_params).reshape(-1,9)
    points_3d = np.array(points_3d)
    points_2d = np.array(points_2d)
    camera_ind = np.array(camera_ind).reshape(len(camera_ind))
    points_ind = np.array(points_ind).reshape(len(points_ind))
    final_l1 = np.array(final_l1)
    final_l2 = np.array(final_l2)
    return cam_params, points_3d, points_2d, camera_ind, points_ind, final_l1, final_l2, low_bound, up_bound, map_des, src_2d


def do_BA2(kp_3d, kp_2d, des, comp_list, H, map_3d, map_2d, map_des, map_cam, map_view, my_update, col, col2, my_max, BA=0):
    """
    Function takes input of current image features, estimated_transformation, map, flags for BA and map update. 
    After processing outputs the updated Map 
    Input:
        kp_3d: 3d_points of current image (nx3)
        kp_2d: 2d_points of current image (nx3)
        des: descriptors (nx32)
        comp_list: Matches array from Flann based matcher 
        map_3d: 3d_points in map (mx3)
        map_2d: Corrosponding 2d_points for map_3d points (m rows and contains set of 3 sized array in each row depend on observations)
        map_des: map_desciptors for 3d_points (mx32)
        map_cam: camera in which map_3d points are viewed (m rows and contains set of 1 sized array in each row depend on observations)
        map_view: contains transformation of the cameras seens (mx6)
        my_update: flag to update map, 0 for not to update; 1 for update 
        col: Colors in map (mx3)
        col2: colors in image (nx3)
        my_max : (no of current camera - 1)
        BA: Flag to perform BA or not
    Output:
        H_op : final transformation (3x4)
        map_3d: 3d_points in map (mx3)
        map_2d: Corrosponding 2d_points for map_3d points (m rows and contains set of 3 sized array in each row depend on observations)
        map_des: map_desciptors for 3d_points (mx32)
        map_cam: camera in which map_3d points are viewed (m rows and contains set of 1 sized array in each row depend on observations)
        map_view: contains transformation of the cameras seens (mx6)
        col: Colors in map (mx3)
        my_max : (no of current camera - 1)
        len(l2): Matches
         
    """
    # Setting the Format of inputs for using BA modules
    camera_params, points_3d, points_2d, camera_ind, points_ind, final_l1, final_l2, low_bound, up_bound, map_des, map_2d = get_things1(kp_3d, kp_2d, des, comp_list, H, map_3d, map_2d, map_des, map_cam, map_view, my_max)
    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]
    n = 9 * n_cameras + 3 * n_points
    m = 2 * points_2d.shape[0]
    # Optimisation Variable
    x0 = np.hstack((camera_params.ravel(), points_3d[:, 0:3].ravel()))
    resx = x0.copy()
    if(BA==1):
        # Standard BA Module
        f0 = fun(x0, n_cameras, n_points, camera_ind, points_ind, points_2d[:,:2], points_2d[:,2])
        A = bundle_adjustment_sparsity(n_cameras, n_points, camera_ind, points_ind)
        t0 = time.time()

        res = least_squares(fun, x0, jac_sparsity=A, bounds=(low_bound, up_bound), verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                            args=(n_cameras, n_points, camera_ind, points_ind, points_2d[:,:2], points_2d[:,2]))
        t1 = time.time()

        resx = res.x
    # Updating the Map with updated points and transformations
    my_min = 0
    my_max = np.max(camera_ind)+1
    H_op = np.zeros((3,4))
    H_op[0:3,0:3] = R.from_rotvec(resx[(my_max-1)*9:(my_max-1)*9+3]).as_matrix()
    H_op[0:3,3] = resx[(my_max-1)*9+3:(my_max-1)*9+6] # Updating the final transformation
    
    final_pts = np.array(resx[my_max*9:]).reshape(-1,3)
    ini_pts = np.array(x0[my_max*9:]).reshape(-1,3)
    map_view = np.vstack((map_view,resx[(my_max-1)*9:(my_max-1)*9+6])) # Updating Transformations in the map

    for i in range(my_min,my_max-1):
        map_view[i] = resx[i*9 : i*9+6]
    update_list = []
    count = 0
    count1 = 0
    for i in range(len(final_l1)):
        # Identifying the Map points
        if(final_l2[i]==1):
            update_list.append(final_l1[i])
        if(final_l2[i]==0):
            count1 += 1
            err = np.sqrt(np.sum(np.square((final_pts[points_ind[i]] - ini_pts[points_ind[i]]).ravel()))/3)
            map_3d[final_l1[i]] = final_pts[points_ind[i]]  # Updating the map points
            if(np.max(map_cam[final_l1[i]])!=my_max-1):
                map_cam[final_l1[i]].append(my_max-1)   # Updating the map views
                count +=1
    
    # Adding the Notseen points to the Map
    update_list = np.array(update_list)
    l2 = np.unique(np.sort(update_list))
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
        new_col = np.array(new_col)
        map_3d = np.vstack((map_3d,new_3d))
        map_des = np.vstack((map_des,new_des))
        col = np.vstack((col,new_col))

    return H_op, map_3d, map_2d, map_des, map_cam, map_view, col, my_max-1, len(l2) 
 

def plot_img(good1,img2,kp_2d2,num, mask2, sc):
    """
    Function takes input of current image, matches, mask, inliers_score, keypoints and Saves the images with matches labelled on them
    Input:
        good1: Matches array of Flann based matcher type array
        img2: image array
        kp_2d2: key_points from ORB format
        mask: obtained mask in numpy array
        sc: score an array of zero or 1 of length as good1 array
    Output: None - but saves the image in the mentioned location 
    """
    k = np.array([kp_2d2[m.trainIdx,0:2] for m in good1])
    img3 = img2.copy()
    for ii in range(len(k)):
        x = int(k[ii,0])
        y = int(k[ii,1])
        img3 = cv2.circle(img2,(x,y),2,(0,255,255),-1)
    filename = './results/f2m/match/'+str(num)+'.jpg'
    cv2.imwrite(filename, img3)  

def write_data1(row):
    with open('data_match.csv', 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(row)
        f_object.close()
    return


init = 0 # Map Initialiser needs to be set zero

# Transformations List
H_ini = []
H_list = []
second_point = 0 
col = []

# Trajectory List
txs = []
tys = []
thetas = []
match_nums = []
not_match_nums = []
map_fea = []

# give the address of the images in the get_imgs functions
i =0
total_images_count = 1100
H_prev = np.array([1,0,0,0,1,0,0,0,1]).reshape(3,3)
x_prev = 0
y_prev = 0
tx_gt = []
ty_gt = []

sample_freq = 5 # rate at which Sampling needs to be done
refresh_rate = 20   # rate at which Map refresh needs to be done

# While loop starts the F2M process
while(i<total_images_count):
    if (init==0 ):
        # Map Initialization
        yes = 0
        first_point = i # First Frame for Map Initialization
        second_point = i+sample_freq # Second frame for map initialisation
        print("image" + str(i) + " is processing")
        
        #Image 1
        img1, mask1, depth1 = get_imgs(i)   # Load images, mask, depth
        kp_3d1,kp_2d1, des1, col1 = get_features(img1,mask1,depth1) # getting ORB features from the img

        # Creating Map Structure
        H_fin = np.array([1,0,0,0,0,1,0,0,0,0,1,0]).reshape(3,4)
        map_view = []
        map_view.append([0,0,0,0,0,0])
        map_view = np.array(map_view).reshape(-1,6)
        map_3d = np.array(kp_3d1)
        map_cam = []
        des_list = []
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
        col = np.array(col1)

        init = 1    # Turn this off to '0' Code will work as f2f
        
        # Image 2
        img2, mask2, depth2 = get_imgs(second_point)    # Load images, mask, depth
        kp_3d2, kp_2d2, des2, col2 = get_features(img2,mask2,depth2) # getting ORB features from the img
        good1 = match_features(map_2d, kp_2d2, map_des, des2)   # matches features and output the matches
        H_2d, score,scale, tx, ty, _ = get_tracking(good1, map_3d, kp_3d2, 0)   # f2m tracking returns 2D transformation
        H_fin = get_3d_H(H_2d)  # Getting 3D transform from 2D transform
        
        #Bundle adjustment
        H_op, map_3d, map_2d, map_des, map_cam, map_view, col, my_max, my_match = do_BA2(kp_3d2, kp_2d2, des2, good1, H_fin, map_3d, map_2d, map_des, map_cam, map_view, 1, col, col2,0,0)
        col = np.array(col)
        final_pts = map_3d
        final_col = col
    
    elif(i%sample_freq==0):
        # Map update 
        yes = 1
        print("image" + str(i) + " is processing")
        
        if(((i%sample_freq)==0) and (i!=second_point) and (i!=first_point)):
            
            img2, mask2, depth2 = get_imgs(i) # Loading the image, mask, depth
            kp_3d2, kp_2d2, des2, col2 = get_features(img2,mask2,depth2) # getting ORB features from the img
            good1 = match_features(map_2d, kp_2d2, map_des, des2) # matches features and output the matches
            H_2d, score,scale, tx, ty, theta = get_tracking(good1, map_3d, kp_3d2, 0) # f2f tracking returns 2D transformation
            plot_img(good1,img2,kp_2d2,i, mask2, score) # plotting the inliers and matches
            H_fin = get_3d_H(H_2d)  # Getting 3D transform from 2D transform

            # Bundle adjustment
            H_op, map_3d, map_2d, map_des, map_cam, map_view, col, my_max, my_match = do_BA2(kp_3d2, kp_2d2, des2, good1, H_fin, map_3d, map_2d, map_des, map_cam, map_view, 1, col, col2,my_max,0)
            H_fin = H_op # Updating the Transformation
            # Storing the Map
            col = np.array(col)
            final_pts = map_3d
            final_col = col

        # Getting the World Transformation 
        H_fin1 = np.linalg.inv(H_fin[0:3,0:3]@H_prev)@(H_fin[:,3])
        H_fin2 = H_fin[0:3,0:3]@H_prev
        theta = np.arctan2(-H_fin2[0,2], H_fin2[0,0])
        tx = -H_fin1[0]
        ty = -H_fin1[2]

        # Trajectory update
        txs.append(tx+x_prev)
        tys.append(ty+y_prev)
        thetas.append(theta)

        #Plotting the trajectory
        plt.axis("equal")
        plt.scatter(txs, tys)
        plt.scatter(txs[0], tys[0], c ='green')
        plt.scatter(txs[-1], tys[-1], c = 'orange')
        plt.savefig("./results/f2m/traj/traj_"+str(i)+".png")
        plt.savefig("./results/f2m/traj/traj.png")
        plt.close("all")
        txs1 = np.array(txs)
        tys1 = np.array(tys)
        thetas1 = np.array(thetas)
        map_fea1 = np.array(map_fea)
        np.save('./results/f2m/x_co1', txs1)
        np.save('./results/f2m/y_co1', tys1)
        np.save('./results/f2m/theta_co1', thetas1)
        
        # For Corrospondence Experiment inside below if condition ; we create a no.of.3d_pts x no.of.cams x 3 array [pixel_location_x, pixel_location_y, Visibility_in_camera(0,255)] 
        if(i%refresh_rate==0):
            take_away=[]
            saver = np.zeros((3,len(map_2d),my_max+1))
            for xx in range(len(map_2d)):
                this_row = map_2d[xx]
                this_cam = map_cam[xx]
                if(len(this_row)>1):
                    take_away.append(xx)
                if(len(this_row)!=len(this_cam)):
                    print("ERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR")
                    break
                for yy in range(len(this_row)):
                    saver[0,xx,this_cam[yy]] = this_row[yy][0]
                    saver[1,xx,this_cam[yy]] = this_row[yy][1]
                    saver[2,xx,this_cam[yy]] = 255
            np.save('./results/f2m/coros/'+str(i)+'_map.npy',saver)
        

        # For Updating the Previous Transformations ; 
        if((i%refresh_rate==0) and (i!=first_point) and (i!=440)):
            init = 0
            H_prev = H_fin[0:3,0:3]@H_prev
            x_prev = tx+x_prev
            y_prev = ty+y_prev
            i -= sample_freq
    i +=sample_freq

txs = np.array(txs)
tys = np.array(tys)
thetas = np.array(thetas)
match_nums = np.array(match_nums)
not_match_nums = np.array(not_match_nums)
map_fea = np.array(map_fea)
np.save('./results/f2m/theta_co', thetas)
np.save('./results/f2m/matches', match_nums)
np.save('./results/f2m/x_co', txs)
np.save('./results/f2m/y_co', tys)
