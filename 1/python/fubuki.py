"""
ELEN0060-2 - Information and coding theory
University of Liège
Academic year 2019-2020

Project 1 - Part 2
Designing informative experiments
"""


###########
# Imports #
###########

import numpy as np


#############
# Functions #
#############

def valid(i, j, grid, clues = None, row_c = None, col_c = None):
	if clues is not None:
		if clues[i, j] != 0 and clues[i, j] != grid[i, j]:
			return False

	# (a)
	for k in range(grid.shape[0]):
		for l in range(grid.shape[1]):
			if k == i and l == j:
				break
			elif grid[k, l] == grid[i, j]:
				return False

		if k == i:
			break

	# (b)
	if row_c is not None:
		if j == grid.shape[1] - 1:
			if grid[i].sum() != row_c[i]:
				return False
		elif grid[i].sum() >= row_c[i]:
			return False

	if col_c is not None:
		if i == grid.shape[0] - 1:
			if grid[:, j].sum() != col_c[j]:
				return False
		elif grid[:, j].sum() >= col_c[j]:
			return False

	return True

def solve(grid, domain, clues = None, row_c = None, col_c = None):
	n, m = grid.shape
	solutions = []

	i, j = 0, 0

	while i >= 0:
		grid[i, j] += 1
		
		while grid[i, j] in domain:
			if valid(i, j, grid, clues, row_c, col_c):
				break
			else:
				grid[i, j] += 1

		if grid[i, j] not in domain:
			grid[i, j] = 0
			
			if j == 0:
				i, j = i - 1, m - 1
			else:
				j -= 1
		elif i == n - 1 and j == m - 1:
			solutions.append(grid.copy())
		else:
			if j == m - 1:
				i, j = i + 1, 0
			else:
				j += 1

	return solutions


########
# Main #
########

if __name__ == '__main__':
	# Fubuki grid

	domain = range(1, 9 + 1)

	clues = np.array([
		[4, 0, 0],
		[0, 0, 0],
		[0, 0, 1]
	])

	row_c = np.array([14, 22, 9])
	col_c = np.array([15, 21, 9])

	# 14. Entropy of subgrid (a)

	solutions = solve(np.zeros((1, 3)), domain, clues)
	N = len(solutions)
	H = np.log2(N)

	print("14. Number of solutions :\n", N)
	print("14. Entropy of subgrid :\n", H)

	# 15. Entropy of subgrid (a) & (b)

	solutions = solve(np.zeros((1, 3)), domain, clues, row_c)
	N = len(solutions)
	H = np.log2(N)

	print("15. Number of solutions :\n", N)
	print("15. Entropy of subgrid :\n", np.log2(N))

	# 17. Entropy of single square (A)

	N = np.zeros(clues.shape)

	for i in range(clues.shape[0]):
		for j in range(clues.shape[1]):
			if clues[i, j] == 0:
				for x in domain:
					if x in clues:
						continue
					elif x + clues[i].sum() > row_c[i]:
						continue
					elif x + clues[:, j].sum() > col_c[j]:
						continue

					N[i, j] += 1
			else:
				N[i, j] = 1

	H = np.log2(N)

	print("17. Number of solutions of single square(s) :\n", N)
	print("17. Entropy of single square(s) :\n", H)

	# 18. Entropy of unsolved Fubuki grid (A)

	print("18. Entropy of grid :\n", H.sum())
