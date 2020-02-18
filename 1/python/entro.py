"""
ELEN0060-2 - Information and coding theory
University of Liège
Academic year 2019-2020

Project 1 - Part 1
Information measures
"""


###########
# Imports #
###########

import numpy as np


#############
# Functions #
#############

def expectation(X, P):
	return (X * P).sum()

def information(p):
	q = p.copy()
	q[p == 0] = 1
	return -np.log2(q)

def entropy(P_X): # 8
	return expectation(information(P_X), P_X)

def joint_entropy(P_XY): # 9
	return entropy(P_XY.reshape(-1))

def conditional_entropy(P_XY): # 10
	P_Y = P_XY.sum(axis=0)
	return expectation(information(P_XY) - information(P_Y), P_XY)

def mutual_information(P_XY): # 11
	P_X = P_XY.sum(axis=1)
	return entropy(P_X) - conditional_entropy(P_XY)

def cond_joint_entropy(P_XYZ): # 12
	P_WZ = P_XYZ.reshape(-1, P_XYZ.shape[2])
	return conditional_entropy(P_WZ)

def cond_mutual_information(P_XYZ): # 12
	P_XZ = P_XYZ.sum(axis=1)
	P_XW = P_XYZ.reshape(P_XYZ.shape[0], -1)
	return conditional_entropy(P_XZ) - conditional_entropy(P_XW)


########
# Main #
########

if __name__ == '__main__':
	# Exercise by hand

	X = np.arange(4, dtype=int)
	P_X = np.array([1/4, 1/4, 1/4, 1/4])
	Y = np.arange(4, dtype=int)
	P_Y = np.array([1/2, 1/4, 1/8, 1/8])
	
	## 1.a. X & Y joint probability distribution

	P_XY = np.outer(P_X, P_Y)

	## 1.b. S & Z marginal probability distribution

	S = np.unique(np.add.outer(X, Y))
	P_S = np.zeros(S.shape)

	Z = np.unique(np.equal.outer(X, Y))
	P_Z = np.zeros(Z.shape)

	for i in np.arange(len(X)):
		for j in np.arange(len(Y)):
			P_S[S == np.add(X[i], Y[j])] += P_XY[i, j]
			P_Z[Z == np.equal(X[i], Y[j])] += P_XY[i, j]

	print("1. XY joint probability distribution :\n", P_XY)
	print("1. S marginal probability distribution :\n", P_S)
	print("1. Z marginal probability distribution :\n", P_Z)

	## 1.c. X, Y, Z & S joint probability distribution

	P_XYSZ = np.zeros((len(X), len(Y), len(S), len(Z)))

	for i in np.arange(len(X)):
		for j in np.arange(len(Y)):
			for k in np.arange(len(S)):
				for l in np.arange(len(Z)):
					if S[k] == np.add(X[i], Y[j]) and Z[l] == np.equal(X[i], Y[j]):
						P_XYSZ[i, j, k, l] = P_XY[i, j]

	P_XS = P_XYSZ.sum(axis=(1, 3))
	P_XZ = P_XYSZ.sum(axis=(1, 2))
	P_YS = P_XYSZ.sum(axis=(0, 3))
	P_YZ = P_XYSZ.sum(axis=(0, 2))
	P_SZ = P_XYSZ.sum(axis=(0, 1))

	P_XYS = P_XYSZ.sum(axis=3)

	## 2. Entropies

	H_X, H_Y, H_S, H_Z = entropy(P_X), entropy(P_Y), entropy(P_S), entropy(P_Z)
	print("2. Marginal entropies :\n", H_X, H_Y, H_S, H_Z)

	## 3. Joint entropies

	H_XY, H_XS, H_YZ, H_SZ = joint_entropy(P_XY), joint_entropy(P_XS), joint_entropy(P_YZ), joint_entropy(P_SZ)
	print("3. Joint entropies :\n", H_XY, H_XS, H_YZ, H_SZ)

	## 4. Conditional entropies

	H_X_Y, H_Z_X, H_S_X, H_S_Z = conditional_entropy(P_XY), conditional_entropy(P_XZ.T), conditional_entropy(P_XS.T), conditional_entropy(P_SZ)
	print("4. Conditional entropies :\n", H_X_Y, H_Z_X, H_S_X, H_S_Z)

	## 5. Conditional joint entropies

	H_XY_S, H_SY_X = cond_joint_entropy(P_XYS), cond_joint_entropy(np.transpose(P_XYS, (2, 1, 0)))
	print("5. Conditional joint entropies :\n", H_XY_S, H_SY_X)

	## 6. Mutual informations

	I_XY, I_XS, I_YZ, I_SZ = mutual_information(P_XY), mutual_information(P_XS), mutual_information(P_YZ), mutual_information(P_SZ)
	print("6. Mutual informations :\n", I_XY, I_XS, I_YZ, I_SZ)

	## 7. Conditional mutual informations

	I_XY_S, I_SY_X = cond_mutual_information(P_XYS), cond_mutual_information(np.transpose(P_XYS, (2, 1, 0)))
	print("7. Conditional mutual informations :\n", I_XY_S, I_SY_X)

	# Computer-aided exercises

	## 13. Samples

	N = 10000

	x = np.random.choice(X, size=N, p=P_X)
	y = np.random.choice(Y, size=N, p=P_Y)
	s = np.add(x, y)
	z = np.equal(x, y)

	xysz = np.column_stack((x, y, s, z))

	## 13.1. Frequence distributions

	u_x, x_counts = np.unique(x, return_counts=True)
	f_x = np.zeros((len(X)))
	for i, (x) in enumerate(u_x):
		f_x[X == x] = x_counts[i] / N

	u_y, y_counts = np.unique(y, return_counts=True)
	f_y = np.zeros((len(Y)))
	for i, (y) in enumerate(u_y):
		f_y[Y == y] = y_counts[i] / N

	u_xy, xy_counts = np.unique(xysz[:, :2], axis=0, return_counts=True)
	f_xy = np.zeros((len(X), len(Y)))
	for i, (x, y) in enumerate(u_xy):
		f_xy[X == x, Y == y] = xy_counts[i] / N

	u_s, s_counts = np.unique(s, return_counts=True)
	f_s = np.zeros((len(S)))
	for i, (s) in enumerate(u_s):
		f_s[S == s] = s_counts[i] / N

	u_z, z_counts = np.unique(z, return_counts=True)
	f_z = np.zeros((len(Z)))
	for i, (z) in enumerate(u_z):
		f_z[Z == z] = z_counts[i] / N

	u_xysz, xysz_counts = np.unique(xysz, axis=0, return_counts=True)
	f_xysz = np.zeros((len(X), len(Y), len(S), len(Z)))
	for i, (x, y, s, z) in enumerate(u_xysz):
		f_xysz[X == x, Y == y, S == s, Z == z] = xysz_counts[i] / N

	f_xs = f_xysz.sum(axis=(1, 3))
	f_xz = f_xysz.sum(axis=(1, 2))
	f_ys = f_xysz.sum(axis=(0, 3))
	f_yz = f_xysz.sum(axis=(0, 2))
	f_sz = f_xysz.sum(axis=(0, 1))

	f_xys = f_xysz.sum(axis=3)

	print("13.1. XY joint frequence distribution :\n", f_xy)
	print("13.1. S marginal frequence distribution :\n", f_s)
	print("13.1. Z marginal frequence distribution :\n", f_z)

	## 13.2. Entropies

	h_x, h_y, h_s, h_z = entropy(f_x), entropy(f_y), entropy(f_s), entropy(f_z)
	print("13.2. Marginal entropies :\n", h_x, h_y, h_s, h_z)

	## 13.3. Joint entropies

	h_xy, h_xs, h_yz, h_sz = joint_entropy(f_xy), joint_entropy(f_xs), joint_entropy(f_yz), joint_entropy(f_sz)
	print("13.3. Joint entropies :\n", h_xy, h_xs, h_yz, h_sz)

	## 13.4. Conditional entropies

	h_x_y, h_z_x, h_s_x, h_s_z = conditional_entropy(f_xy), conditional_entropy(f_xz.T), conditional_entropy(f_xs.T), conditional_entropy(f_sz)
	print("13.4. Conditional entropies :\n", h_x_y, h_z_x, h_s_x, h_s_z)

	## 13.5. Conditional joint entropies

	h_xy_s, h_sy_x = cond_joint_entropy(f_xys), cond_joint_entropy(np.transpose(f_xys, (2, 1, 0)))
	print("13.5. Conditional joint entropies :\n", h_xy_s, h_sy_x)

	## 13.6. Mutual informations

	i_xy, i_xs, i_yz, i_sz = mutual_information(f_xy), mutual_information(f_xs), mutual_information(f_yz), mutual_information(f_sz)
	print("13.6. Mutual informations :\n", i_xy, i_xs, i_yz, i_sz)

	## 13.7. Conditional mutual informations

	i_xy_s, i_sy_x = cond_mutual_information(f_xys), cond_mutual_information(np.transpose(f_xys, (2, 1, 0)))
	print("13.7. Conditional mutual informations :\n", i_xy_s, i_sy_x)
