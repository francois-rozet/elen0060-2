#!/usr/bin/env python

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

def create(row = None, column = None):
	'''Creates a ruler that filters out invalid values for a square.'''

	if row is None and column is None:
		def ruler(clues, values, index):
			return list(values)
	else:
		def aux(row, values, constraint):
			y = constraint - row.sum()

			if (row == 0).sum() == 1:
				# sum of row equals constraint
				return [y] if y in values else []
			else:
				# sum of row is lower than constraint
				return [x for x in values if x <= y]

		if column is None:
			def ruler(clues, values, index):
				return aux(clues[index[0]], values, row[index[0]])
		elif row is None:
			def ruler(clues, values, index):
				return aux(clues[:, index[1]], values, column[index[1]])
		else:
			def ruler(clues, values, index):
				# successive filters
				return aux(clues[:, index[1]], aux(clues[index[0]], values, row[index[0]]), column[index[1]])

	return ruler


def solve(grid, domain, ruler):
	'''Generates all solutions matching the rules for the current grid.'''

	# Initialize
	unused = set([x for x in domain if x not in grid])
	free = set([index for (index , x) in np.ndenumerate(grid) if x == 0])
	guesses = []

	# First guess
	guesses.append(None)

	for index in free:
		values = ruler(grid, unused, index)
		if guesses[-1] is None or len(guesses[-1][1]) > len(values):
			guesses[-1] = (index, values)

	free.remove(guesses[-1][0])

	# Until all guesses have been consumed
	while guesses:
		if grid[guesses[-1][0]] != 0:
			unused.add(grid[guesses[-1][0]])

		if guesses[-1][1]:
			grid[guesses[-1][0]] = guesses[-1][1].pop()
			unused.remove(grid[guesses[-1][0]])

			# If all squares are filled
			if not free:
				yield grid.copy()
				continue

			# New guess
			guesses.append(None)

			for index in free:
				values = ruler(grid, unused, index)
				if guesses[-1] is None or len(guesses[-1][1]) > len(values):
					guesses[-1] = (index, values)

			free.remove(guesses[-1][0])
		else:
			# Backtrack
			free.add(guesses[-1][0])
			grid[guesses[-1][0]] = 0
			guesses.pop()


########
# Main #
########

if __name__ == '__main__':
	# 14. Entropy of subgrid (a)

	domain = range(1, 9 + 1)

	clues = np.array([
		[4, 0, 0]
	])
	row_constraints = np.array([14])

	ruler = create()

	solutions = list(solve(clues, domain, ruler))
	N = len(solutions)
	H = np.log2(N)

	print('14. Number of solutions :\n', N)
	print('14. Entropy of subgrid :\n', H)

	# 15. Entropy of subgrid (a) & (b)

	ruler = create(row_constraints)

	solutions = list(solve(clues, domain, ruler))
	N = len(solutions)
	H = np.log2(N)

	print('15. Number of solutions :\n', N)
	print('15. Entropy of subgrid :\n', np.log2(N))

	# 17. Entropy of single square (A)

	clues = np.array([
		[4, 0, 0],
		[0, 0, 0],
		[0, 0, 1]
	])
	row_constraints = np.array([14, 22, 9])
	column_constraints = np.array([15, 21, 9])

	unused = [x for x in domain if x not in clues]

	ruler = create(row_constraints, column_constraints)

	N = np.zeros(clues.shape)
	for index, x in np.ndenumerate(clues):
		if x == 0:
			N[index] = len(ruler(clues, unused, index))
		else:
			N[index] = 1

	H = np.log2(N)

	print('17. Number of solutions of single square(s) :\n', N)
	print('17. Entropy of single square(s) :\n', H)

	# 18. Entropy of unsolved Fubuki grid (A)

	print('18. Entropy of grid :\n', H.sum())

	# 19. Without assumption A

	solutions = list(solve(clues, domain, create(row_constraints, column_constraints)))
	N = len(solutions)
	H = np.log2(N)

	print('19. All solutions :\n', solutions)
	print('19. Number of solutions :\n', N)
	print('19. Entropy of grid :\n', H)
