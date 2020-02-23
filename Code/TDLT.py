import glob
import cv2
import numpy as np 
import random

def TDLT(ca, h4pt):
	A = []
	B = []

	for i in range(4):
		x, y = ca[i][0], ca[i][1]
		u, v = x+h4pt[i], y+h4pt[i+5]
		B.append(-v)
		B.append(u)
		tempA1 = [0 0 0 -x -y -1 x*v y*v]
		tempA2 = [x y 1 0 0 0 -x*u -y*u]
		A.append(tempA1)
		A.append(tempA2)

	Ainv = np.linalg.pinv(np.array(A))
	H = np.matmul(Ainv, np.array(B))
	return H


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

# Form the equations Ax = b to compute H
# Form A matrix
A1 = tf.matmul(M1_tile, pts_1_tile) # Column 1
A2 = tf.matmul(M2_tile, pts_1_tile) # Column 2
A3 = M3_tile                   # Column 3
A4 = tf.matmul(M4_tile, pts_1_tile) # Column 4
A5 = tf.matmul(M5_tile, pts_1_tile) # Column 5
A6 = M6_tile                   # Column 6
A7 = tf.matmul(M71_tile, pred_pts_2_tile) *  tf.matmul(M72_tile, pts_1_tile)# Column 7
A8 = tf.matmul(M71_tile, pred_pts_2_tile) *  tf.matmul(M8_tile, pts_1_tile)# Column 8

A_mat = tf.transpose(tf.stack([tf.reshape(A1,[-1,8]),tf.reshape(A2,[-1,8]),\
                               tf.reshape(A3,[-1,8]),tf.reshape(A4,[-1,8]),\
                               tf.reshape(A5,[-1,8]),tf.reshape(A6,[-1,8]),\
     tf.reshape(A7,[-1,8]),tf.reshape(A8,[-1,8])],axis=1), perm=[0,2,1]) # BATCH_SIZE x 8 (A_i) x 8
print('--Shape of A_mat:', A_mat.get_shape().as_list())
# Form b matrix
b_mat = tf.matmul(Mb_tile, pred_pts_2_tile)
print('--shape of b:', b_mat.get_shape().as_list())

# Solve the Ax = b
H_8el = tf.matrix_solve(A_mat , b_mat)  # BATCH_SIZE x 8.
print('--shape of H_8el', H_8el)


# Add ones to the last cols to reconstruct H for computing reprojection error
h_ones = tf.ones([batch_size, 1, 1])
H_9el = tf.concat([H_8el,h_ones],1)
H_flat = tf.reshape(H_9el, [-1,9])
self.H_mat = tf.reshape(H_flat,[-1,3,3])   # BATCH_SIZE x 3 x 3
