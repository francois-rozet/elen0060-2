#!/usr/bin/env python

"""
ELEN0060-2 - Information and coding theory
University of Liège
Academic year 2019-2020

Project 2 - Part 1
Source coding and reversible data compression
"""


###########
# Imports #
###########

import numpy as np
import utils

from heapq import heappush, heappop, heapify
from math import ceil, log2


#############
# Functions #
#############

def huffman_tree(source, distribution, alphabet=['1', '0']):
	'''Returns an Huffman tree of a source given its distribution.'''

	# Initialize
	tree = PriorityQueue(list(zip(
		distribution,
		map(lambda x: id(x), source),
		source
	)))

	# Until all branches are merged
	while len(tree) > 1:
		p, subtree = 0, {}

		for letter in alphabet:
			if not tree:
				break

			# Pop lowest probability branch
			x, _, branch = tree.pop()

			# Aggregate probability and assign a letter
			p += x
			subtree[letter] = branch

		# Push subtree as new branch
		tree.push((p, id(subtree), subtree))

	return tree.pop()[-1]


def huffman_code(tree):
	'''Returns the Huffman code associated to an Huffman tree.'''

	# Initialize
	code = {}

	queue = [('', tree)]

	# Until queue is empty
	while queue:
		# Pop subtree
		prefix, subtree = queue.pop()

		for letter, branch in subtree.items():
			if type(branch) is not dict:
				# If branch is leaf, add codeword to code
				code[branch] = prefix + letter
			else:
				# Else, push branch to queue with new prefix
				queue.append((prefix + letter, branch))

	return code


def lempel_ziv(stream):
	'''On-line basic Lempel-Ziv encoder.'''

	# Initialize dictionary
	code = {}
	nbits = 0
	prefix, address = '', 0

	# Until end of input stream
	for symbol in stream:
		prefix += symbol

		## Search in dictionary
		if prefix in code:
			address = code[prefix]
			continue

		## Output
		for x in utils.int_to_bin(address, nbits)[:nbits]:
			yield x
		yield symbol

		## Update dictionary
		code[prefix] = len(code) + 1
		nbits = len(code).bit_length()
		prefix, address = '', 0

	# Remaining suffix
	if prefix:
		for x in utils.int_to_bin(address, nbits):
			yield x


def lempel_ziv_inv(stream):
	'''On-line Lempel-Ziv decoder.'''

	# Initialize decoded list
	code = ['']
	nbits = 0
	prefix, address = '', ''

	# Until end of input stream
	for symbol in stream:
		if len(address) < nbits:
			address += symbol
			continue

		## Search in decoded list
		if address:
			prefix = code[utils.bin_to_int(address)]

		## Output
		for x in prefix:
			yield x
		yield symbol

		## Update decoded list
		nbits = len(code).bit_length()
		code.append(prefix + symbol)
		prefix, address = '', ''

	# Remaining suffix
	if address:
		prefix = code[utils.bin_to_int(address)]
		for x in prefix:
			yield x


###########
# Classes #
###########

class PriorityQueue:
	def __init__(self, items=None):
		if items is None:
			self.heap = []
		else:
			self.heap = items
			heapify(self.heap)

	def __bool__(self):
		return bool(self.heap)

	def __len__(self):
		return len(self.heap)

	def push(self, item):
		heappush(self.heap, item)

	def pop(self):
		return heappop(self.heap)


class Encoder:
	def __init__(self, code):
		self.code = code

	def __call__(self, stream):
		for symbol in stream:
			yield(self.code[symbol])


class Decoder:
	def __init__(self, tree):
		self.tree = tree

	def __call__(self, stream):
		subtree = self.tree

		for symbol in stram:
			if type(subtree[symbol]) is dict:
				subtree = subtree[symbol]
			else:
				yield subtree[symbol]
				subtree = self.tree


########
# Main #
########

if __name__ == '__main__':
	# Parameters

	origin = '../resources/'

	# Source coding and reversivle data compression

	text = utils.load(origin + 'text.txt')
	byte_text = utils.load_byte(origin + 'text.txt', spaces=False)

	## 1. Set of source symbols

	S, counts = np.unique(list(text), return_counts=True)
	Q = len(S)

	print('1. Set of source symbols :\n', S)
	print('1. Q :\n', Q)

	## 2. Marginal probability distribution

	P = counts / counts.sum()
	H = - np.dot(P, np.log2(P))

	print('2. Marginal probability distribution :\n', P)

	## 4. Huffman encode

	q = 2

	code = huffman_code(huffman_tree(S, P))
	encoder = Encoder(code)

	encoded_text = ''.join(encoder(text))

	print('4. Huffman code :', *code.items(), sep='\n')
	print('4. Encoded text length :\n', len(encoded_text))
	print('4. Empirical average length :\n', len(encoded_text) / len(text))

	## 5. Huffman expected length

	expected = 0

	for symbol, codeword in code.items():
		expected += P[np.argmax(S == symbol)] * len(codeword)

	print('5. Expected average length :\n', expected)
	print('5. Lower bound on average length :\n', H / np.log2(q))

	## 6. Compression rate

	print('6. Compression rate :\n', np.log2(Q) / expected)

	## 11. Lempel-Ziv

	encoded_byte_text = ''.join(lempel_ziv(byte_text))
	decoded_byte_text = ''.join(lempel_ziv_inv(encoded_byte_text))

	print('11. Byte text length :\n', len(byte_text))
	print('11. Encoded byte text length :\n', len(encoded_byte_text))
	print('11. Decoded byte text length :\n', len(decoded_byte_text))
