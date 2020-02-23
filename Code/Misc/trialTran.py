import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import cv2
from TFSpatialTransformer import *
from utils import *


def DLT(pts_1_tile, pred_pts_2_tile):
	batch_size = 1
	M1 = tf.constant(Aux_M1, tf.float32)
	M1_tensor = tf.expand_dims(M1, [0])
	M1_tile = tf.tile(M1_tensor,[batch_size,1,1])
	
	M2 = tf.constant(Aux_M2, tf.float32)
	M2_tensor = tf.expand_dims(M2, [0])
	M2_tile = tf.tile(M2_tensor,[batch_size,1,1])

	M3 = tf.constant(Aux_M3, tf.float32)
	M3_tensor = tf.expand_dims(M3, [0])
	M3_tile = tf.tile(M3_tensor,[batch_size,1,1])

	M4 = tf.constant(Aux_M4, tf.float32)
	M4_tensor = tf.expand_dims(M4, [0])
	M4_tile = tf.tile(M4_tensor,[batch_size,1,1])

	M5 = tf.constant(Aux_M5, tf.float32)
	M5_tensor = tf.expand_dims(M5, [0])
	M5_tile = tf.tile(M5_tensor,[batch_size,1,1])

	M6 = tf.constant(Aux_M6, tf.float32)
	M6_tensor = tf.expand_dims(M6, [0])
	M6_tile = tf.tile(M6_tensor,[batch_size,1,1])

	M71 = tf.constant(Aux_M71, tf.float32)
	M71_tensor = tf.expand_dims(M71, [0])
	M71_tile = tf.tile(M71_tensor,[batch_size,1,1])

	M72 = tf.constant(Aux_M72, tf.float32)
	M72_tensor = tf.expand_dims(M72, [0])
	M72_tile = tf.tile(M72_tensor,[batch_size,1,1])

	M8 = tf.constant(Aux_M8, tf.float32)
	M8_tensor = tf.expand_dims(M8, [0])
	M8_tile = tf.tile(M8_tensor,[batch_size,1,1])

	Mb = tf.constant(Aux_Mb, tf.float32)
	Mb_tensor = tf.expand_dims(Mb, [0])
	Mb_tile = tf.tile(Mb_tensor,[batch_size,1,1])

	A1 = tf.matmul(M1_tile, pts_1_tile) # Column 1
	A2 = tf.matmul(M2_tile, pts_1_tile) # Column 2
	A3 = M3_tile                   # Column 3
	A4 = tf.matmul(M4_tile, pts_1_tile) # Column 4
	A5 = tf.matmul(M5_tile, pts_1_tile) # Column 5
	A6 = M6_tile                   # Column 6
	A7 = tf.matmul(M71_tile, pred_pts_2_tile) *  tf.matmul(M72_tile, pts_1_tile) # Column 7
	A8 = tf.matmul(M71_tile, pred_pts_2_tile) *  tf.matmul(M8_tile, pts_1_tile)  # Column 8

	A_mat = tf.transpose(tf.stack([tf.reshape(A1,[-1,8]),tf.reshape(A2,[-1,8]),\
								   tf.reshape(A3,[-1,8]),tf.reshape(A4,[-1,8]),\
								   tf.reshape(A5,[-1,8]),tf.reshape(A6,[-1,8]),\
								   tf.reshape(A7,[-1,8]),tf.reshape(A8,[-1,8])],
								   axis=1), perm=[0,2,1]) # BATCH_SIZE x 8 (A_i) x 8
	print('--Shape of A_mat:', A_mat.get_shape().as_list())
	b_mat = tf.matmul(Mb_tile, pred_pts_2_tile)
	print('--shape of b:', b_mat.get_shape().as_list())

	# Solve the Ax = b
	H_8el = tf.matrix_solve(A_mat , b_mat)  # BATCH_SIZE x 8.
	print('--shape of H_8el', H_8el)
	
	# Add ones to the last cols to reconstruct H for computing reprojection error
	h_ones = tf.ones([batch_size, 1, 1])
	H_9el = tf.concat([H_8el,h_ones], 1)
	H_flat = tf.reshape(H_9el, [-1,9])
	H_mat = tf.reshape(H_flat, [-1,3,3])   # BATCH_SIZE x 3 x 3
	return H_mat

im1 = cv2.imread("1_1.jpg", 0)
w, h =128,128
xy = np.array([[0,0],[128,0],[128,128],[0,128]],dtype="f")
xy_ = np.array([[10,0],[138,0],[128,128],[0,128]],dtype="f")

uv = np.array([0,0,128,0,128,128,0,128])
del_uv = np.array([10,0,10,0,0,0,0,0])

# uv = np.array([0,128,128,0,0,0,128,128])
# del_uv = np.array([10,10,0,0,0,0,0,0])

uv_new = uv + del_uv
# H = np.array([[1.093,6.93e-02,-3.36e+01],[1.642e-02,1.11e+00,-2.58e+02],[1.446e-02,2.618e-04,1.0]],dtype="f")
# H = np.array([[1,0,0],[0,1,0],[0,0,1]])
# HT = tf.convert_to_tensor(H, dtype=tf.float32)
img = tf.convert_to_tensor(im1, dtype=tf.float32)
img = tf.expand_dims(img,2)
img = tf.expand_dims(img,0)
out_size = tf.convert_to_tensor(np.array([128,128]),dtype=tf.int32)

pts_1_tile = tf.convert_to_tensor(uv, dtype=tf.float32)
pts_1_tile = tf.expand_dims(pts_1_tile, 0)
pts_1_tile = tf.expand_dims(pts_1_tile, 2)
pred_pts_2_tile = tf.convert_to_tensor(uv_new, dtype=tf.float32)
pred_pts_2_tile = tf.expand_dims(pred_pts_2_tile, 0)
pred_pts_2_tile = tf.expand_dims(pred_pts_2_tile, 2)
H = DLT(pts_1_tile, pred_pts_2_tile)

M = np.array([[w/2 , 0 , w/2],
			  [ 0 , h/2 , h/2],
			  [ 0 ,  0 ,  1]])

M_inv = np.linalg.inv(M)

M_t = tf.convert_to_tensor(M,dtype=tf.float32)
# M_t = tf.expand_dims(M_t,2)
M_t = tf.expand_dims(M_t,0)
M_inv_t = tf.convert_to_tensor(M_inv, dtype=tf.float32)
# M_inv_t = tf.expand_dims(M_inv_t,2)
M_inv_t = tf.expand_dims(M_inv_t,0)
H_ = tf.matmul(tf.matmul(M_inv_t, H), M_t)

output, condition = transformer(img, H_, out_size)
with tf.Session() as sess:
	l,h = sess.run([output,H])

print("-------------")
print(h)
print("---From opencv----")
print(cv2.getPerspectiveTransform(xy,xy_))
plt.imshow(l[0,:,:,0], cmap="gray")
plt.show()
