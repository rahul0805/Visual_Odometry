import sys
sys.path.remove(sys.path[1])

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse


def FP(f, cx, cy, h, imWidth, imHeight):
	depth = np.zeros((int(imHeight), int(imWidth)), np.float32)
	for i in range(int(cy)+1, int(imHeight)):
		depth[i,:] = f*h/(i-cy)
	return depth




def convert(points2D, f, cx, cy):
	"""
    Function: Generates 3D points from 2D pixels
    Input: 2D Image pixels, depth(cms), focal length, cx,cy 
    Method: Perspective projection of PAL Camera
    Output: 3D_points in meters 
    """
	num_pts = points2D.shape[0]
	us = points2D.reshape(-1,2)[:,0]
	vs = points2D.reshape(-1,2)[:,1]
	Zs = depth[vs.astype(int),us.astype(int)]
	thetas = (us / (2*cx)).astype(int) * np.pi / 3 
	us_ = us.copy()
	us = us % (2*cx)		
	Xs = (us-cx)*Zs/f
	
	new_Xs = Xs * np.cos(thetas) + Zs * np.sin(thetas) 
	new_Zs = -Xs * np.sin(thetas) + Zs * np.cos(thetas)
	points3D = np.zeros((num_pts,2), np.float32)
	points3D[:,0] = new_Xs.ravel()
	points3D[:,1] = new_Zs.ravel()
	points3D = points3D.reshape(-1,1,2)
	return points3D


def estimate_pure_rotation(img1, img2):
	"""
    Function takes input of images and outputs the  rotated theta. 
    Input:
        img1: first image
		img2: Second image
    Output:
        theta: rotated angle

    """
	h,w,c = img1.shape
	gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	
	templ = cv2.resize(gray1, (w//4,h//4))
	templ = templ[150:350, 250:650]
	
	dst = cv2.resize(gray2, (w//4,h//4))
	
	res = cv2.matchTemplate(dst,templ,cv2.TM_SQDIFF)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	top_left = min_loc
	
	disp = top_left[0] - 250
	f = w//4 / 12 *3**0.5
	
	theta = np.arctan(disp / f)
		
	return theta

def plot_img(good1,img2,kp_2d2, sc):
    k = np.array([kp_2d2[m.trainIdx].pt for m in good1])
    img3 = img2.copy()
    for ii in range(len(k)):
        x = int(k[ii,0])
        y = int(k[ii,1])
        img3 = cv2.circle(img2,(x,y),2,(0,255,255),-1)
        if((sc[ii]==[1])):
            img3 = cv2.circle(img2,(x,y),2,(0,0, 255),-1)
    filename = './results/f2f/match/'+str(fname)+'.jpg'
    cv2.imwrite(filename, img3)  

def estimate_pure_translation(img1, mask1, img2, mask2, theta, count):
	"""
    Function takes input of images and Floor masks and outputs the  score, tx,ty, theta. 
    Input:
        img1: first image
		img2: Second image
		Mask1: Floor mask 1
		Mask2: Floor mask 2
		Theta: Estimated theta (if calculated previously)
		Count: Current Frame number 
    Output:
        score: To determine no of inliers
        tx: translation in x
        ty: translation in y
        theta: rotated angle

    """
	img3 = img2.copy()
	img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	img1 = clahe.apply(img1)
	img2 = clahe.apply(img2)
	kp1, des1 = orb.detectAndCompute(img1, mask=mask1)
	kp2, des2 = orb.detectAndCompute(img2, mask=mask2)
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

	src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
	dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

	src_3d_pts =  convert(src_pts, f, cx, cy)
	dst_3d_pts =  convert(dst_pts, f, cx, cy)

	H, score = cv2.estimateAffinePartial2D(src_3d_pts/100, dst_3d_pts/100, ransacReprojThreshold=0.050)
	theta = np.arctan2(H[0,1], H[0,0])
	scale = H[0,0] / np.cos(theta)
	tx = H[0,2]
	ty = H[1,2]	
	return tx, ty, np.sum(score), theta

cy = 670/4
height = 65.5
imWidth = 3584/4
imHeight  = 1218/4
cx = imWidth / 12
f = (cx * 3**0.5)
	

depth = FP(f, cx, cy, height, imWidth, imHeight)
orb = cv2.ORB_create(nfeatures = 200000)
H_prev = np.eye(3)
txs = []
tys = []
thetas = []
stamps = []
nFeatures = []
img1 =  None
mask1 = None

clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
    
fname = 0
while (fname<357):
	if (fname % 1 != 0):
		continue

	img2 = cv2.imread("/home/rahul/dreamvu/visual_odo/dataset/carpet_1/{}.png".format(fname))
	mask2 = cv2.imread("/home/rahul/dreamvu/visual_odo/dataset/carpet_1/train_floor/{}.png".format(fname), 0)



	if img2 is None or mask2 is None:
		break	

	img2 = cv2.resize(img2, (int(imWidth), int(imHeight)))
	mask2 = cv2.resize(mask2, (int(imWidth), int(imHeight)))
	mask2 = cv2.inRange(mask2, 150, 255, cv2.THRESH_BINARY)
	mask2 = cv2.dilate(mask2, np.ones((15,15)))


	if img1 is None:
		img1 = img2.copy()
		mask1 = mask2.copy()
		continue


	# theta = estimate_pure_rotation(img1, img2)
	tx, ty, num_features, theta = estimate_pure_translation(img1, mask1, img2, mask2, 0, fname)
	if num_features:
		new_H = np.eye(3);
		new_H[0,0] = np.cos(theta)
		new_H[0,1] = np.sin(theta)
		new_H[0,2] = tx
		new_H[1,0] = -np.sin(theta)
		new_H[1,1] = np.cos(theta)
		new_H[1,2] = ty
		temp = new_H @ H_prev 
		final_H = np.linalg.inv(temp)
		theta = np.arctan2(-final_H[0,1], final_H[0,0]) 
		tx = final_H[0,2]	
		ty = final_H[1,2]
		H_prev = temp.copy();
		txs.append(tx)
		tys.append(ty)
		thetas.append(theta)
		nFeatures.append(num_features)
		# stamps.append(timestamps[fname])
		print("img_num : {}, nfeatures : {}, tx : {}, ty : {}, theta : {} ".format(fname,  num_features, tx*100, ty*100, int(theta/np.pi*180)))
		img1 = img2.copy()
		mask1 = mask2.copy()
		np.save('/results/f2f/x_co_f2f', txs)
		np.save('/results/f2f/y_co_f2f', tys)
		np.save('/results/f2f/theta_co_f2f', thetas)
		plt.axis("equal")
		plt.scatter(txs, tys)
		plt.savefig("./results/f2f/f2f.png")
		plt.close("all")
	fname +=1
		
