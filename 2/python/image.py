#!/usr/bin/env python

"""
ELEN0060-2 - Information and coding theory
University of Liège
Academic year 2019-2020

Project 2 - Part 2
Reversible image compression
"""


###########
# Imports #
###########

import numpy as np
import utils

from text import Huffman, LZ78


########
# Main #
########

if __name__ == '__main__':
	# Parameters

	origin = '../resources/'

	# Reversible image compression

	image = utils.load_mat(origin + 'lena512.mat')['lena512']
	shape = image.shape
	image = image.ravel()

	"""
	## Filtering
	image[1:] -= image[:-1]
	image %= 256
	## Unfiltering
	image = np.cumsum(image) % 256
	"""

	## Byte encoding

	byte_image = [utils.int_to_bin(x, n=8) for x in image]

	print('13. Byte image length :\n', sum(map(len, byte_image)))

	## 13. Huffman

	S, counts = np.unique(byte_image, axis=0, return_counts=True)
	P = counts / counts.sum()

	tree = Huffman.tree(S.tolist(), P)
	code = Huffman.code(tree)

	encoded_byte_image = ''.join(Huffman.encode(code, byte_image))
	decoded_byte_image = ''.join(Huffman.decode(tree, encoded_byte_image))

	print('13. Huffman encoded image length :\n', len(encoded_byte_image))
	print('13. Huffman decoded image length :\n', len(decoded_byte_image))

	## 13. LZ78 and Huffman

	indexes = []
	symbols = []

	for index, symbol in LZ78.encode(byte_image, size=None):
		indexes.append(index)
		symbols.append(symbol)

	### Indexes

	S, counts = np.unique(indexes, axis=0, return_counts=True)
	P = counts / counts.sum()

	index_tree = Huffman.tree(S, P)
	index_code = Huffman.code(index_tree)

	index_encoder = Huffman.encode(index_code, indexes)

	### Symbols

	S, counts = np.unique(symbols, axis=0, return_counts=True)
	P = counts / counts.sum()

	symbol_tree = Huffman.tree(S, P)
	symbol_code = Huffman.code(symbol_tree)

	symbol_encoder = Huffman.encode(symbol_code, symbols)

	### Combining

	encoded_byte_image = ''
	for index, symbol in zip(index_encoder, symbol_encoder):
		encoded_byte_image += index + symbol

	print('13. LZ78 + Huffman encoded byte image length :\n', len(encoded_byte_image))

	stream = iter(encoded_byte_image)
	index_decoder = Huffman.decode(index_tree, stream)
	symbol_decoder = Huffman.decode(symbol_tree, stream)

	decoded_byte_image = zip(index_decoder, symbol_decoder)
	decoded_byte_image = ''.join(''.join(x) for x in LZ78.decode(decoded_byte_image, size=None))

	print('13. LZ78 + Huffman decoded byte image length :\n', len(decoded_byte_image))
