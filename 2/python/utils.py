#!/usr/bin/env python

"""
ELEN0060-2 - Information and coding theory
University of Liège
Academic year 2019-2020

Project 2 - Utils
"""


###########
# Imports #
###########

import numpy as np
from scipy.io import loadmat, wavfile


#############
# Functions #
#############

def int_to_bin(value, n=0):
	'''Returns the n-bits binary representation of a decimal value.'''
	return np.binary_repr(value, width=n)


def bin_to_int(b):
	'''Returns the decimal value of a binary representation.'''
	return int(b, 2)


def load(filename):
	'''Returns the content of a (text) file.'''
	with open(filename, 'r') as f:
		return f.read()


def str_to_byte(s):
	'''Returns a string as a byte stream.'''
	return map(lambda x: int_to_bin(ord(x), 8), s)


def load_byte(filename, spaces=True):
	'''Returns the content of (text) file as bytes.'''
	inter = ' ' if spaces else ''
	return inter.join(str_to_byte(load(filename)))


def load_mat(filename):
	'''Loads a MATLAB mat file.'''
	return loadmat(filename)


def load_wav(filename):
	'''Loads a wav file as a sound signal.'''
	return wavfile.read(filename)


def save_wav(filename, rate, data):
	'''Encodes a sound signal as a wav file.'''
	return wavfile.write(filename, rate, data)
