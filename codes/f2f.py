import numpy as np 
import cv2
import open3d as o3d
import os
from matplotlib import pyplot as plt
import random
from csv import writer
from scipy.spatial.transform import Rotation as R
from scipy.sparse import lil_matrix
import time
from scipy.optimize import least_squares

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
    verts = np.hstack([verts[0:-1:100], colors[0:-1:100]])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


# Images location
left_dir = './rgb/' 
img_name = sorted(os.listdir(left_dir))
right_dir = './depth/'
depth_name = sorted(os.listdir(right_dir))
# print(img_name)
new_d = []
new_im = []
c = 0
with open("sync_odom.txt") as fp:
    Lines = fp.readlines()
    while c in range(len(Lines)):
        line = Lines[c]
        new_im.append(line.split()[1])
        new_d.append(line.split()[3])
        c += 1

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
    # thetas = theta * np.pi / 3 
    # points_proj = np.copy(points_proj1)
    # points_proj[:,0] = points_proj1[:,0]*np.cos(thetas) - points_proj1[:,2]*np.sin(thetas)
    # points_proj[:,2] = points_proj1[:,0]*np.sin(thetas) + points_proj1[:,2]*np.cos(thetas)
    cx = 319.5  # optical center x
    cy = 239.5  # optical center y
    points_proj = np.copy(points_proj1)
    points_proj[:,0] = points_proj1[:,0]
    points_proj[:,2] = points_proj1[:,2]
    count = 0
    for i in range(len(points_proj)):
        if(points_proj[i,2]==0):
            points_proj[i,0] = 0
            points_proj[i,1] = 0
            points_proj[i,2] = 1
            count += 1
    # print("Count is: ", count)
    points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj**2, axis=1) #n = (r_c)^2
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, np.newaxis]
    # points_proj[:,0] += cx
    # points_proj[:,1] += cy
    return points_proj



def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d, theta):
    """Compute residuals.
    
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    # print("Point indices here: ", point_indices)
    # print("camera indices here: ", camera_indices)
    # print("camera Params: ", camera_params)
    points_proj = project(points_3d[point_indices], camera_params[camera_indices], theta)
    # print("Residual here:", (points_proj - points_2d).ravel())
    x = ((points_proj - points_2d).ravel())
    # print("Residual is: ", x.T@x)
    return ((points_proj - points_2d).ravel())


def reproj(my_pts):
    fx = 525.0  # focal length x
    fy = 525.0  # focal length y
    cx = 319.5  # optical center x
    cy = 239.5  # optical center y
    img_pts = []
    for i in range(len(my_pts)):
        X = my_pts[i,0]
        Y = my_pts[i,1]
        Z = my_pts[i,2]
        x = (fx * (X/Z)) + cx
        y = (fy * (Y/Z)) + cy
        img_pts.append([x,y])
    img_pts = np.array(img_pts)
    return img_pts



def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2 
    
    n = n_cameras * 9 + n_points * 3 
    
    A = lil_matrix((m, n), dtype=float)

    i = np.arange(camera_indices.size)
    for s in [1,3,5]:
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1

    # for s in [0,2]:
    #     A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
    #     A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1
        
    return A

def call_BA(camera_params, points_3d, points_2d, camera_ind, points_ind, final_l1, final_l2, low_bound, up_bound, map_des, map_2d):
    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]
    n = 9 * n_cameras + 3 * n_points
    m = 2 * points_2d.shape[0]

    x0 = np.hstack((camera_params.ravel(), points_3d[:, 0:3].ravel()))
    resx = x0.copy()
    # print(x0)
    print(x0.shape)
    f0 = fun(x0, n_cameras, n_points, camera_ind, points_ind, points_2d[:,:2], points_2d[:,2])
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_ind, points_ind)
    t0 = time.time()

    res = least_squares(fun, x0, jac_sparsity=A, bounds=(low_bound, up_bound), verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                        args=(n_cameras, n_points, camera_ind, points_ind, points_2d[:,:2], points_2d[:,2]))
    t1 = time.time()

    resx = res.x
    return resx

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
    f = 525.0
    for i in range(my_min,my_max+1):
        cam_param = [map_view[i,0], map_view[i,1], map_view[i,2], map_view[i,3], map_view[i,4], map_view[i,5], f,0,0]
        cam_params.append(cam_param)
        low_bound.append(-np.pi)
        low_bound.append(-np.pi)
        low_bound.append(-np.pi)
        low_bound.append(-0.1)
        low_bound.append(-np.inf)
        low_bound.append(-0.1)
        low_bound.append(f-1)
        low_bound.append(-1)
        low_bound.append(-1)
        up_bound.append(np.pi)
        up_bound.append(np.pi)
        up_bound.append(np.pi)
        up_bound.append(0.1)
        up_bound.append(np.inf)
        up_bound.append(0.1)
        up_bound.append(f)
        up_bound.append(0)
        up_bound.append(0)
    r = (R.from_matrix((H[0:3, 0:3]))).as_rotvec()
    t = H[:,3]
    for i in range(3):
        if(abs(t[i])>0.1):
            t[i]=0
    cam_param = [r[0], r[1], r[2], t[0], t[1], t[2], f, 0, 0]
    cam_params.append(cam_param)
    # print(cam_params)
    
    low_bound.append(-np.pi)
    low_bound.append(-np.pi)
    low_bound.append(-np.pi)
    low_bound.append(-0.1)
    low_bound.append(-np.inf)
    low_bound.append(-0.1)
    low_bound.append(f-1)
    low_bound.append(-1)
    low_bound.append(-1)
    up_bound.append(np.pi)
    up_bound.append(np.pi)
    up_bound.append(np.pi)
    up_bound.append(0.1)
    up_bound.append(np.inf)
    up_bound.append(0.1)
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
    # l_fin = l[:,l[1, :].argsort()]
    l_fin = l
    j = 0
    count = len(points_3d)
    prev = -1
    final_l1 = []
    final_l2 = []
    final_des = []
    fx = 525.0  # focal length x
    fy = 525.0  # focal length y
    cx = 319.5  # optical center x
    cy = 239.5  # optical center y
    while(j<(len(l_fin[0]))):
        i1 = l_fin[0,j]
        i2 = l_fin[1,j]
        if(i2!=prev):
            check = 0
            m_2d = src_2d[i1]
            check = 1
            ind = int(0)
            points_2d.append([int((m_2d[0])-cx), int((m_2d[1])-cy),0])
            points_ind.append(count)
            camera_ind.append(ind)
            final_l1.append(i1)
            final_l2.append(0)
            if(check==1):
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
                # src_2d[i1].append([int((dst_2d[i2,0])), int((dst_2d[i2,1]))])
        j+=1
    cam_params = np.array(cam_params).reshape(-1,9)
    points_3d = np.array(points_3d)
    points_2d = np.array(points_2d)
    camera_ind = np.array(camera_ind).reshape(len(camera_ind))
    points_ind = np.array(points_ind).reshape(len(points_ind))
    final_l1 = np.array(final_l1)
    final_l2 = np.array(final_l2)
    return cam_params, points_3d, points_2d, camera_ind, points_ind, final_l1, final_l2, low_bound, up_bound, map_des, src_2d




def convert_3d(points_2d, depth_image, image):
    fx = 525.0  # focal length x
    fy = 525.0  # focal length y
    cx = 319.5  # optical center x
    cy = 239.5  # optical center y
    factor = 5208 # for the 16-bit PNG files
    points = []
    colors = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for i in range(len(points_2d)):
        x = int(points_2d[i,0])
        y = int(points_2d[i,1])
        # print(y)
        Z = depth_image[y,x] / factor
        X = (x - cx) * Z / fx
        Y = (y - cy) * Z / fy
        points.append([X,Y,Z])
        colors[y,x] = [255,0,0]
    points_3d = []
    cols = []
    for v in range(depth_image.shape[0]):
        for u in range(depth_image.shape[1]):
            Z = depth_image[v,u] / factor
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            points_3d.append([X,Y,Z])
            cols.append(colors[v,u])
    
    points_3d = np.array(points_3d)
    cols = np.array(cols)
    points = np.array(points)
    
    return points, points_3d, cols

            

def plot_img(good1,img2,kp_2d2, sc):
    k = kp_2d2
    img3 = img2.copy()
    for ii in range(len(k)):
        x = int(k[ii,0])
        y = int(k[ii,1])
        img3 = cv2.circle(img2,(x,y),2,(0,255,255),-1)
    filename = './match_'+str(fname)+'.jpg'
    cv2.imwrite(filename, img3)  

def get_3d_H(H1):
    H_fin = [H1[0,0], 0, H1[0,1], H1[0,2], 0, 1, 0, 0, H1[1,0], 0, H1[1,1], H1[1,2]]
    H_fin = np.array(H_fin).reshape(3,4)
    return H_fin

def estimate_pure_rotation(img1, img2):
    h,w,c = img1.shape
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    templ = gray1[150:330,200:440]
    dst = gray2[150:330,200:440]
    res = cv2.matchTemplate(gray2,templ,cv2.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = min_loc
    disp = top_left[0] - 200 #change here
    # print(top_left[0])
    f = 525.0
    theta = np.arctan(disp / f)
    return theta

def estimate_pure_translation(img1, mask1, depth1, img2, mask2, depth2):
    f = 525.0
    img3 = img1.copy()
    img4 = img2.copy()
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1 = clahe.apply(img1)
    img2 = clahe.apply(img2)
    img1 = cv2.GaussianBlur(img1,(5,5),0)
    img2 = cv2.GaussianBlur(img2,(5,5),0)
    kp1, des1 = orb.detectAndCompute(img1, mask=mask1)
    kp2, des2 = orb.detectAndCompute(img2, mask=mask2)
    # for i in range(len(kp1)):
    #     print(kp1[i].size)
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
        if m.distance < 0.8*n.distance:
            good.append(m)

    if  len(good) < 20:
        return 0., 0., 0., 0.
    # for m in good:
    #     print(kp1[m.queryIdx].response)
    # kp_1 = []
    # kp_2 = []
    # # for i in range(len(kp1)):
    kp_1 = np.float32([ kp1[i].pt for i in range(len(kp1)) ]).reshape(-1,2)
    kp_2 = np.float32([ kp2[i].pt for i in range(len(kp2)) ]).reshape(-1,2)

    src_3d_pts1, src_3d_gt, src_col  =  convert_3d(kp_1, depth1, img3)
    dst_3d_pts1, dst_3d_gt, dst_col =  convert_3d(kp_2, depth2, img4)

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,2)

    # src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,2)
    # dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,2)
    # row = [fname, len(dst_pts), len(kp2)]
    # write_data1(row)
    
    src_3d_pts, src_3d_gt, src_col  =  convert_3d(src_pts, depth1, img3)
    dst_3d_pts, dst_3d_gt, dst_col =  convert_3d(dst_pts, depth2, img4)
    # col = np.array(col)
    # final_pts = src_3d_pts
    # final_col = np.zeros((src_3d_pts.shape))
    final_pts = src_3d_gt
    final_col = src_col
    file_name = './results/pcd/map_0_'+str(fname)+'_10.ply'
    # write_ply(file_name,final_pts,final_col)
    diff = ((src_pts-dst_pts)**2).reshape(-1,2)
    dist = np.sqrt(diff[:,0]+diff[:,1])
    print("Calculated Pixel mean and median of distance: ",np.mean(dist), np.median(dist))
    
    src_2d = []
    dst_2d = []
    sr = []
    ds = []
    for i in range(len(src_pts)):
        sr.append([src_3d_pts[i,0], src_3d_pts[i,2]])
        ds.append([dst_3d_pts[i,0], dst_3d_pts[i,2]])
        if((abs(src_3d_pts[i,1]-dst_3d_pts[i,1])<0.1)):
            src_2d.append([src_3d_pts[i,0], src_3d_pts[i,2]])
            dst_2d.append([dst_3d_pts[i,0], dst_3d_pts[i,2]])

    src_2d = np.array(src_2d).reshape(-1,2)
    dst_2d = np.array(dst_2d).reshape(-1,2)
    sr = np.array(sr).reshape(-1,2)
    ds = np.array(ds).reshape(-1,2)
    diff = ((sr-ds)**2).reshape(-1,2)
    # print(src_2d)
    # print(dst_2d)
    dist = np.sqrt(diff[:,0]+diff[:,1])
    H, score = cv2.estimateAffinePartial2D(src_2d, dst_2d, ransacReprojThreshold=0.1)
    # plot_img(good,img4,dst_pts, score)
    # print(H)
    H_3d = get_3d_H(H)
    H_fin = get_3d_H(H)
    s = np.ones((src_3d_pts.shape[0],4))
    s[:,:3] = src_3d_pts
    d = (H_fin@s.T).T
    diff = ((d-dst_3d_pts)**2).reshape(-1,3)
    dist = np.sqrt(diff[:,0]+diff[:,1]+diff[:,2])
    print("Mean 3D reprojection error: ", np.mean(dist))
    cam_params = []
    for i in range(len(d)):
        cam_params.append([0,0,0,0,0,0,f,0,0])
    cam_params = np.array(cam_params).reshape(-1,9)
    d_2d = reproj(d)
    diff = ((d_2d-dst_pts)**2).reshape(-1,2)
    dist1 = np.sqrt(diff[:,0]+diff[:,1])
    # print("2D reprojection error: ", (dist1))
    # print("SUM 2D reprojection error: ", np.median(dist1))
    # print("SUM 2D reprojection error: ", np.sum(dist1))
    print("Mean 2D reprojection error: ", np.mean(dist1))
    prev = np.mean(dist1)
    
    
    if(fname!=-1):
        print(H_3d)
        camera_params, points_3d, points_2d, camera_ind, points_ind, final_l1, final_l2, low_bound, up_bound, map_des, map_2d = get_things1(dst_3d_pts1, kp_2, des2, good, H_3d, src_3d_pts1, kp_1, des1, 0, np.array([[0,0,0,0,0,0,f,0,0]]), 0)
        resx = call_BA(camera_params, points_3d, points_2d, camera_ind, points_ind, final_l1, final_l2, low_bound, up_bound, map_des, map_2d)
        H_op1 = np.eye(4)
        H_op2 = np.zeros((3,4))
        #H_op = np.array(resx[(my_max-1)*9:(my_max-1)*9+6])
        my_max = 1
        H_op1[0:3,0:3] = R.from_rotvec(resx[(my_max-1)*9:(my_max-1)*9+3]).as_matrix()
        H_op1[0:3,3] = resx[(my_max-1)*9+3:(my_max-1)*9+6]
        # print(H_op)
        my_max = 2
        H_op2[0:3,0:3] = R.from_rotvec(resx[(my_max-1)*9:(my_max-1)*9+3]).as_matrix()
        H_op2[0:3,3] = resx[(my_max-1)*9+3:(my_max-1)*9+6]
        H_op = H_op2@(np.linalg.inv(H_op1))
        print(H_op)
        # H = H_op
        s = np.ones((src_3d_pts.shape[0],4))
        s[:,:3] = src_3d_pts
        d = (H_op@s.T).T
        diff = ((d-dst_3d_pts)**2).reshape(-1,3)
        dist = np.sqrt(diff[:,0]+diff[:,1]+diff[:,2])
        print("Mean 3D reprojection error: ", np.mean(dist))
        cam_params = []
        for i in range(len(d)):
            cam_params.append([0,0,0,0,0,0,f,0,0])
        cam_params = np.array(cam_params).reshape(-1,9)
        d_2d = reproj(d)
        diff = ((d_2d-dst_pts)**2).reshape(-1,2)
        dist1 = np.sqrt(diff[:,0]+diff[:,1])
        # print("2D reprojection error: ", (dist1))
        # print("SUM 2D reprojection error: ", np.median(dist1))
        # print("SUM 2D reprojection error: ", np.sum(dist1))
        print("Mean 2D reprojection error: ", np.mean(dist1))
        now = np.mean(dist1)
        if(prev>now):
            H = np.zeros((2,3))
            H[0,0] = H_op[0,0]
            H[0,1] = H_op[0,2]
            H[1,0] = H_op[2,0]
            H[1,1] = H_op[2,2]
            H[0,2] = H_op[0,3]
            H[1,2] = H_op[2,3]
    theta = np.arctan2(H[0,1], H[0,0])
    scale = H[0,0] / np.cos(theta)
    tx = H[0,2]
    ty = H[1,2]
    est_move = np.sqrt(tx**2+ty**2)
	# print("Lenght of less estimateion ", len(samp_dist), len(diff))
    print("Calculated mean and median of distance: ",np.mean(dist), np.median(dist))	
    # print("Scale is: ", scale)
    # print(np.sum(score))
    return tx, ty, theta, H, src_3d_gt, dst_3d_gt, src_col, dst_col, np.sum(score), scale


def write_data1(row):
    with open('./results/data_match_f2f.csv', 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(row)
        f_object.close()
    return
def get_3d_H(H1):
    H_fin = [H1[0,0], 0, H1[0,1], H1[0,2], 0, 1, 0, 0, H1[1,0], 0, H1[1,1], H1[1,2]]
    H_fin = np.array(H_fin).reshape(3,4)
    return H_fin

orb = cv2.ORB_create(nfeatures = 1000)
# img1 = cv2.imread(left_dir + img_name[1607])   #107 
# img2 = cv2.imread(left_dir + img_name[1617])   #112 #1607

# depth1 = cv2.imread(right_dir + depth_name[1590], -1)
# depth2 = cv2.imread(right_dir + depth_name[1600], -1)

# mask1 = ((depth1>0)).astype(int)*255
# mask2 = ((depth2>0)).astype(int)*255
# mask1 = cv2.inRange(mask1, 150, 255, cv2.THRESH_BINARY)
# mask2 = cv2.inRange(mask2, 150, 255, cv2.THRESH_BINARY)
# tx, ty, theta, H, src_3d_pts, dst_3d_pts, src_col, dst_col = estimate_pure_translation(img1, mask1, depth1, img2, mask2, depth2)
# H_fin = get_3d_H(H)
# print(src_3d_pts.shape)
# s = np.ones((src_3d_pts.shape[0],4))
# s[:,:3] = src_3d_pts
# d = (H_fin@s.T).T
# print(d.shape)
# final_pts = np.vstack((d,dst_3d_pts))
# final_col = np.vstack((src_col, dst_col))
# file_name = './results/pcd/f2f.ply'
# write_ply(file_name,final_pts,final_col)

# # H_fin[0,3] = -0.15
# # H_fin[2,3] = -0.09
# theta_gt = -0.085
# H_fin[0,0] = np.cos(theta_gt)
# H_fin[2,2] = np.cos(theta_gt)
# H_fin[0,2] = np.sin(theta_gt)
# H_fin[2,0] = -np.sin(theta_gt)
# H_fin[0,3] = -0.09
# H_fin[2,3] = -0.15
# s = np.ones((src_3d_pts.shape[0],4))
# s[:,:3] = src_3d_pts
# d = (H_fin@s.T).T
# print(d.shape)
# final_pts = np.vstack((d,dst_3d_pts))
# final_col = np.vstack((src_col, dst_col))
# file_name = './results/pcd/gt.ply'
# write_ply(file_name,final_pts,final_col)
    
# print(tx, ty, theta)
img1 = None
depth1 = None
mask1 =None
H_prev = np.eye(3)
tys = []
txs = []
thetas = []
vel_x = []
vel_y = []
print(new_im[1])
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
fname = 60
flag = 0
f_ini = fname
while (fname < 900):
    # print(fname)
    if (fname%1!= 0):
        continue
    # if (fname==130):
    #     print(left_dir + new_im[fname])


    img2 = cv2.imread('./' + new_im[fname])
    depth2 = cv2.imread('./' + new_d[fname], -1)
    if img2 is None or depth2 is None:
        print("Hello")
        print(left_dir + new_im[fname])
        break	

    mask2 = ((depth2>0)).astype(int)*255
    mask2 = cv2.inRange(mask2, 150, 255, cv2.THRESH_BINARY)

    if img1 is None:
        img1 = img2.copy()
        depth1 = depth2.copy()
        mask1 = mask2.copy()
        continue


    theta1 = estimate_pure_rotation(img1, img2)
    
    tx, ty, theta, H, src_3d_pts, dst_3d_pts, src_col, dst_col, num_features, scale = estimate_pure_translation(img1, mask1, depth1, img2, mask2, depth2)
    print(theta1)
    print(theta)
    # col = np.array(col)
    # final_pts = src_3d_pts
    # final_col = col
    # file_name = './reults/pcd/map_0_'+str(i)+'_10.ply'
    # write_ply(file_name,final_pts,final_col)
                
    if(abs(scale-1)>0.3):
        tx = 0
        ty = 0
        theta = theta1
    
    # if(abs(theta1-theta)>0.3):
    #     tx = 0
    #     ty = 0
    # if(abs(theta1-theta)>0.01):
    #     theta = theta1
    
        # tx = 0
        # ty = 0
    
    # print(tx,ty, theta, theta1)
    if num_features:
        new_H = np.eye(3)
        new_H[0,0] = np.cos(theta)
        new_H[0,1] = np.sin(theta)
        new_H[0,2] = tx
        new_H[1,0] = -np.sin(theta)
        new_H[1,1] = np.cos(theta)
        new_H[1,2] = ty
        this_theta = theta
        temp = new_H@H_prev 
        final_H = np.linalg.inv(temp)
        H = get_3d_H(final_H)
        theta = np.arctan2(-final_H[0,1], final_H[0,0]) 
        tx = final_H[0,2]	
        ty = final_H[1,2]
        H_prev = temp.copy()
        c =  0.05762176
        if(flag==0):
            prev_vel = np.array([0,0]) 
            prev_pt = np.array([tx, ty])
            # flag =1
        else:
            pt = np.array([tx, ty])
            vel = pt-prev_pt
            vel_x.append((prev_pt-c*np.sin(theta))[0])
            vel_y.append((prev_pt+c*np.cos(theta))[1])
            print("Velocites are here: ")
            print(vel, prev_vel)
            x = (prev_pt-c*np.sin(theta))[0]
            y = (prev_pt+c*np.cos(theta))[1]
            d = np.sqrt((tx-x)**2 + (ty-y)**2)
            c1 = -0.05762176
            x1 = (prev_pt-c1*np.sin(theta))[0]
            y1 = (prev_pt+c1*np.cos(theta))[1]
            d1 = np.sqrt((tx-x1)**2 + (ty-y1)**2)
            
            print("D values are here: ",d,d1)
            print(tx, ty, x,y,x1,y1)
            if(((d>0.01) and (d1>0.01))):
                
                # if(fname>500 and fname<800):
                if((d>=d1)):
                    tx = x1
                    ty = y1
                    final_H[0,2] = tx
                    final_H[1,2] = ty
                    x = x1
                    y = y1  
                else:
                # if(fname>0):
                    tx = x
                    ty = y
                    final_H[0,2] = tx
                    final_H[1,2] = ty
            # if(((d>0.01) or (d1>0.01)) and (d1<d)):
            #     tx = x1
            #     ty = y1
            #     final_H[0,2] = tx
            #     final_H[1,2] = ty
            else:
                x = tx
                y = ty
            prev_vel = vel
            prev_pt = np.array([x,y])
        H = get_3d_H(final_H)
        txs.append(tx)
        tys.append(ty)
        thetas.append(theta)
        if(flag==0):
            flag = 1
            final_pts = src_3d_pts  
            final_col = src_col
        else:
            s = np.ones((dst_3d_pts.shape[0],4))
            s[:,:3] = dst_3d_pts
            d2 = (H@s.T).T
            final_pts = np.vstack((final_pts,d2[0:-1:10]))
            final_col = np.vstack((final_col, dst_col[0:-1:10]))
        

        
        # print(tx,ty, theta, theta1)
        
        print("img_num : {}, nfeatures : {}, tx : {}, ty : {}, theta : {} ".format(fname,  num_features, tx, ty, int(theta/np.pi*180)))
        img1 = img2.copy()
        mask1 = mask2.copy()
        depth1 = depth2.copy()
        # np.save('tx_'+str(f_ini), txs)
        # np.save('ty_'+str(f_ini), tys)
        # np.save('thetas_'+str(f_ini), thetas)
        # col = np.array(col)
        # final_pts = src_3d_pts
        # final_col = col
        # file_name = './results/pcd/map_BA_test.ply'
        # write_ply(file_name,final_pts,final_col)
        file_name = './map_test.ply'
        write_ply(file_name,final_pts,final_col)
        # if(fname%500==0):
            # final_col = col
        # val = input("Enter your value: ")
        # if(val=='n'):
        plt.axis("equal")
        plt.scatter(txs, tys)
        # plt.scatter(vel_x, vel_y)
        plt.savefig("./results/f2f_test.png")
        plt.close("all")
        # if(fname==820):
        #     print("Image num is: ", fname)
        # #     cv2.imwrite('this_img.png', img2)
        # #     cv2.imwrite('prev_img.png', img1)
        # #     file_name = './results/pcd/map_BA_just.ply'
        # #     write_ply(file_name,final_pts,final_col)
        #     break
            
    # if (fname==140):
    #     img1 = None
    #     depth1 = None
    #     mask1 = None
    #     fname = 240
    fname += 5

    
