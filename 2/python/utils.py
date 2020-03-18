 #!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from scipy.io import loadmat


"""
This function gives the binary representation of value v (dec)
	using nb bits (0 by default corresponds to the minimal number of
	bits required to represent v).

	examples:
	binarize(2)
	>>> '10'

	binarize(2,nb=8)
	>>> '00000010'

"""
def binarize(v, nb=0):
    if nb == 0:
        return bin(v)[2:]
    else:
        return np.binary_repr(v,width=nb)

# This functions returns the decimal representation of a sequence (str) of bits b
def bin_to_dec(b):
    return int(b,2)

# This function loads the text sample
def load_text_sample():
	f = open('text.csv', 'r')
	return f.read()

# This function loads the text sample and outputs the binary version (8-bits representation).
# If spaces=True, then each byte is separated by a space
def load_binary_text_sample(spaces=True):
	f = open('text.csv', 'r')
	contents = f.read()
	binary_text = ''
	for c in contents:
		if spaces:
			binary_text = binary_text + " " + binarize(ord(c),nb=8)
		else:
			binary_text = binary_text + binarize(ord(c),nb=8)
	return binary_text[1:]


# This function loads the image and returns a matrix of values between [0,255].
def load_image(show=False):
	matrix = loadmat('lena512.mat')
	return np.asarray(matrix)

# This function loads the sound signal (.wav)
def load_wav():
	rate, data = read('sound.wav')
	return rate, data

# This function save the sound signal (.wav)
def save_wav(filename,rate,data):
    scipy.io.wavfile.write(filename, rate, data)
