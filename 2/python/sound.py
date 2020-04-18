#!/usr/bin/env python

"""
ELEN0060-2 - Information and coding theory
University of Liège
Academic year 2019-2020

Project 2 - Part 3
Channel coding
"""


###########
# Imports #
###########

import numpy as np
import os
import random
import utils

from matplotlib import rc
from matplotlib import pyplot as plt


#############
# Functions #
#############

def power_of_two(x):
	return (x & (x - 1) == 0) and x > 0


def channel(signal, p=0.01):
	return ''.join(
		('0' if int(bit) else '1') if random.random() < p else bit
		for bit in signal
	)


###########
# Classes #
###########

class Hamming:
	'''Hamming (n, m) encoder-decoder.'''

	def __init__(self, n=7, m=4):
		# (n, m) in (2 ** r - 1, 2 ** r - r - 1)
		# i.e. (3, 1), (7, 4), (15, 11), (31, 26), ...
		assert n == 2 ** (n - m) - 1

		self.m = m
		self.n = n

		# Parity-check matrix
		self.H = np.ones((n - m, n), dtype=int)

		for i in range(self.n):
			self.H[:, i] = np.array([
				int(bit)
				for bit in utils.int_to_bin(i + 1, n - m)
			], dtype=int)

		# Generator & reversor matrices
		self.G = np.zeros((n, m), dtype=int)
		self.R = np.zeros((m, n), dtype=int)

		for i in range(self.n):
			if power_of_two(i + 1):
				self.G[i, :] = np.array([
					self.H[n - m - (i + 1).bit_length(), j]
					for j in range(n) if not power_of_two(j + 1)
				], dtype=int)
			else:
				j = i - i.bit_length()

				self.G[i, j] = 1
				self.R[j, i] = 1

	def encode(self, stream):
		# Initialize
		word = np.zeros(self.m, dtype=int)
		i = 0

		for bit in stream:
			word[i] = int(bit)
			i += 1

			if i < self.m:
				continue
			else:
				i = 0

			codeword = np.dot(self.G, word) % 2
			yield ''.join(map(str, codeword))

	def decode(self, stream):
		# Initialize
		codeword = np.zeros(self.n, dtype=int)
		i = 0

		for bit in stream:
			codeword[i] = int(bit)
			i += 1

			if i < self.n:
				continue
			else:
				i = 0

			syndrom = np.dot(self.H, codeword) % 2
			e = utils.bin_to_int(''.join(map(str, syndrom)))

			if e:
				codeword[e - 1] = 1 - codeword[e - 1]

			word = np.dot(self.R, codeword) % 2
			yield ''.join(map(str, word))


########
# Main #
########

if __name__ == '__main__':
	# Parameters

	rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
	rc('text', usetex=True)
	plt.rcParams.update({'font.size': 16})

	origin = '../resources/'
	destination = '../products/'

	os.makedirs(destination, exist_ok=True)

	# Channel coding

	hz, sound = utils.load_wav(origin + 'sound.wav')
	time = np.arange(len(sound)) / hz

	## 15. Plot

	plt.figure()
	plt.plot(time, sound, linewidth=0.5)
	plt.xlabel('Time [s]')
	plt.ylabel('Sound')
	plt.tight_layout()
	plt.savefig(destination + 'soundwave.pdf', transparent=True)
	plt.close()

	## 16. Fixed-length encoding

	n = max(int(x).bit_length() for x in sound)

	signal = ''.join(utils.int_to_bin(x, n) for x in sound)

	## 17. Channel noise

	noisy_signal = channel(signal)
	noisy_sound = np.array([
		utils.bin_to_int(noisy_signal[i:i+n])
		for i in range(0, len(noisy_signal), n)
	], dtype=np.uint8)

	plt.figure()
	plt.plot(time, noisy_sound, linewidth=0.5)
	plt.xlabel('Time [s]')
	plt.ylabel('Sound')
	plt.tight_layout()
	plt.savefig(destination + 'noisy_soundwave.pdf', transparent=True)
	plt.close()

	utils.save_wav(destination + 'noisy_sound.wav', hz, noisy_sound)

	## 18. Hamming (7, 4) code

	hamming = Hamming(7, 4)

	encoded_signal = ''.join(hamming.encode(signal))
	decoded_signal = ''.join(hamming.decode(encoded_signal))

	decoded_sound = np.array([
		utils.bin_to_int(decoded_signal[i:i+n])
		for i in range(0, len(decoded_signal), n)
	], dtype=np.uint8)

	utils.save_wav(destination + 'decoded_sound.wav', hz, decoded_sound)

	print('18. Signal length :\n', len(signal))
	print('18. Encoded signal length :\n', len(encoded_signal))
	print('18. Decoded signal length :\n', len(decoded_signal))

	## 19. Channel noise

	noisy_encoded_signal = channel(encoded_signal)
	noisy_decoded_signal = ''.join(hamming.decode(noisy_encoded_signal))

	noisy_decoded_sound = np.array([
		utils.bin_to_int(noisy_decoded_signal[i:i+n])
		for i in range(0, len(noisy_decoded_signal), n)
	], dtype=np.uint8)

	plt.figure()
	plt.plot(time, noisy_decoded_sound, linewidth=0.5)
	plt.xlabel('Time [s]')
	plt.ylabel('Sound')
	plt.tight_layout()
	plt.savefig(destination + 'noisy_decoded_soundwave.pdf', transparent=True)
	plt.close()

	utils.save_wav(destination + 'noisy_decoded_sound.wav', hz, noisy_decoded_sound)
