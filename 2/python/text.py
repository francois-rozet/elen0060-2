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

def pairstr(x):
	return str(x[0]) + ':' + str(x[1])


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


class SlidingWindow:
	def __init__(self, size=256):
		self.__size = size
		self.__window = [None] * self.__size
		self.__cursor = 0

	def __len__(self):
		if self.__window[self.__cursor] is None:
			return self.__cursor
		else:
			return self.__size

	def __geti(self, distance):
		assert distance < self.__size

		i = self.__cursor - distance
		return i if i >= 0 else i + self.__size

	def __getitem__(self, distance):
		return self.__window[self.__geti(distance)]

	def slice(self, distance, length):
		i = self.__geti(distance)

		affix = [None] * length

		k = 0
		for j in range(length):
			if (i + k) % self.__size == self.__cursor:
				k = 0

			affix[j] = self.__window[(i + k) % self.__size]
			k += 1

		return affix

	def push(self, affix):
		for x in affix:
			self.__window[self.__cursor] = x
			self.__cursor += 1
			self.__cursor %= self.__size


class Huffman:
	@staticmethod
	def tree(source, distribution, alphabet=['1', '0']):
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

	@staticmethod
	def code(tree):
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

	@staticmethod
	def encode(code, stream):
		'''Encodes a stream using an Huffman code.'''
		for symbol in stream:
			yield(code[symbol])

	@staticmethod
	def decode(tree, stream):
		'''Decodes a stream using an Huffman tree.'''
		subtree = tree

		for symbol in stream:
			if type(subtree[symbol]) is dict:
				subtree = subtree[symbol]
			else:
				yield subtree[symbol]
				subtree = tree


class LZ78:
	@staticmethod
	def encode(stream, size=None):
		'''On-line basic LZ78 encoder.'''

		# Initialize dictionary
		dictionary = {}
		next_index = 1
		index = 0

		# Until end of input stream
		for symbol in stream:
			## Search in dictionary
			if (index, symbol) in dictionary:
				index = dictionary[(index, symbol)]
				continue

			## Output
			yield index, symbol

			## New dictionary entry
			if size is None or len(dictionary) < size:
				dictionary[(index, symbol)] = next_index
				next_index += 1

			index = 0

		# Remaining suffix
		yield index, ''

	@staticmethod
	def decode(stream, size=None):
		'''On-line basic LZ78 decoder.'''

		# Initialize dictionary
		dictionary = {}
		next_index = 1
		index = 0

		# Until end of input stream
		for index, symbol in stream:
			## Build prefix with dictionary
			prefix = LZ78.build(dictionary, index)

			## Output
			yield prefix + [symbol]

			## Update dictionary
			if size is None or len(dictionary) < size:
				dictionary[next_index] = (index, symbol)
				next_index += 1

			index = 0

		# Remaining suffix
		yield LZ78.build(dictionary, index)

	@staticmethod
	def build(dictionary, index):
		prefix = []
		while index > 0:
			index, symbol = dictionary[index]
			prefix.append(symbol)
		return prefix[::-1]


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

	tree = Huffman.tree(S, P)
	code = Huffman.code(tree)
	encoded_text = ''.join(Huffman.encode(code, text))
	decoded_text = ''.join(Huffman.decode(tree, encoded_text))

	print('4. Huffman code :', *code.items(), sep='\n')
	print('4. Text length :\n', len(text))
	print('4. Encoded text length :\n', len(encoded_text))
	print('4. Decoded text length :\n', len(decoded_text))
	print('4. Empirical average length :\n', len(encoded_text) / len(text))
	print('4. Kraft inequality :', sum(q ** -len(x) for x in code.values()))

	## 5. Huffman expected length

	expected = 0

	for symbol, codeword in code.items():
		expected += P[np.argmax(S == symbol)] * len(codeword)

	print('5. Expected average length :\n', expected)
	print('5. Lower bound on average length :\n', H / np.log2(q))

	## 6. Compression rate

	print('6. Compression rate :\n', np.log2(Q) / expected)

	## 11. Lempel-Ziv

	print('11. Byte text length :\n', len(byte_text))

	encoded_byte_text = list(LZ78.encode(byte_text))

	encoded_len = sum(
		i.bit_length() + len(symbol)
		for i, (index, symbol) in enumerate(encoded_byte_text)
	)

	decoded_byte_text = ''.join(''.join(x) for x in LZ78.decode(encoded_byte_text))

	print('11. LZ78 encoded byte text length :\n', encoded_len)
	print('11. LZ78 decoded byte text length :\n', len(decoded_byte_text))
