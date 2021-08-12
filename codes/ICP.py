
import sys
sys.path.remove(sys.path[1])

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

import open3d as o3d
import copy


my_x = np.load("../x_co1.npy")
my_y = np.load("../y_co1.npy")
my_theta = np.load("../theta_co1.npy")


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    pcds = source_temp
    source_temp.transform(transformation)
    pcd_combined = o3d.geometry.PointCloud()
    pcd_combined = target_temp
    pcds.transform(transformation)
    pcd_combined += pcds
    # o3d.visualization.draw_geometries([pcd_combined])
    return pcd_combined

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])

def read_pcd(i2, target):
    i = int((i2-6)/3)
    source = o3d.io.read_point_cloud("../pcd/map_0_"+str(i2)+"_10.ply")
    print(i)
    H = np.eye(4)
    H[0,0] = np.cos(my_theta[2+(i)*3])
    H[0,2] = np.sin(my_theta[2+(i)*3])
    H[2,0] = -np.sin(my_theta[2+(i)*3])
    H[2,2] = np.cos(my_theta[2+(i)*3])
    H[0,3] = my_x[2+(i)*3]
    H[2,3] = my_y[2+(i)*3]
    source.transform(H)
    
    # print(H)
    trans_init = (np.eye(4))

    # initial_pcd = draw_registration_result(source, target, trans_init)
    
    voxel_radius = [0.04, 0.02, 0.01]
    max_iter = [5000]
    current_transformation = trans_init
    print(current_transformation)
    # # print("3. Colored point cloud registration")
    for scale in range(1):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        # print([iter, radius, scale])

        print("3-1. Downsample with a voxel size %.2f" % radius)
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)
        cl, ind = source_down.remove_statistical_outlier(nb_neighbors=50,
                                                    std_ratio=1.5)
        
        # cl, ind = source_down.remove_radius_outlier(nb_points=10, radius=0.2)
        source_down = source_down.select_by_index(ind)
        # cl, ind = target_down.remove_radius_outlier(nb_points=10, radius=0.2)
        cl, ind = target_down.remove_statistical_outlier(nb_neighbors=50,
                                                    std_ratio=1.5)
        target_down = target_down.select_by_index(ind)
        #display_inlier_outlier(target_down, ind)

        # draw_registration_result(source, target, current_transformation)
        print("3-2. Estimate normal.")
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius*2, max_nn=10))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius*2, max_nn=10))

        print("3-3. Applying colored point cloud registration")
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation, criteria= o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-2,relative_rmse=1e-2  ,max_iteration=iter))
        current_transformation = result_icp.transformation
    print(current_transformation)
    final_pcd = draw_registration_result(source_down, target_down, current_transformation)
    
    return final_pcd, current_transformation

target = o3d.io.read_point_cloud("../pcd/map_0_3_10.ply")
updates = []
for i in range(0,117):
    target, update = read_pcd(6+i*3, target)
    updates.append(update)

target_down = target.voxel_down_sample(0.06)
o3d.io.write_point_cloud("final_ICP.ply", target)
o3d.io.write_point_cloud("final_ICP_down.ply", target_down)
