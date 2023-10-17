import skimage.io as io
import numpy as np
from matplotlib import pyplot as plt
import random
from argparse import ArgumentParser

# global variables
x = 100
y = 100

def recombine(im1 : np.ndarray,
   im2 : np.ndarray
	) -> np.ndarray:
	"""Create a new image from two images. 

	Vars:
		im1: the first image
		im2: the second image

	Returns:
		A new image, chosen by first, randomly choosing
		between the horizontal or vertical orientation,
		and then slicing each image into two pieces along
		a randomly-chosen vertical or horizontal line.
	"""
	output = np.zeros((im1.shape[0], im1.shape[1], 3))
	if (random.random() < .5):
		ind = im1.shape[0]
		slice = random.randint(0, ind)
		output[0 : slice] = im1[0 : slice]
		output[slice : ind] = im2[slice : ind]
	else:
		ind = im1.shape[1]
		slice = random.randint(0, ind)
		output[:, 0 : slice] = im1[:, 0 : slice]
		output[:, slice : ind] = im2[:, slice : ind]
	return output
	

def mutate(im : np.ndarray) -> np.ndarray:
	"""Mutate an image.

	Vars:
		im: the image to mutate.

	Returns:
		A new image, which is the same as the original,
		except that one of the colors is the image is
		globally (i.e., everywhere it occurs in the image)
		replace with a randomly chosen new color.
	"""
	# generates a random color
	color = np.random.rand(3)

	# converts im into an array of 3-length arrays
	im2 = im.reshape(-1, 3)
	# catalogs every unique array in im2
	uniquities = np.unique(im2, axis=0)
	# selects a random element from uniquities
	k = random.randint(0, len(uniquities) - 1)
	modColor = uniquities[k]
	# replaces every instance of modColor with color in im
	b = (im == modColor)
	b = b.reshape((b.shape[0] * b.shape[1], 3))
	b = np.all(b, axis=1)
	b = b.reshape((im.shape[0], im.shape[1]))
	
	im[np.isin(b, True)] = color
	return (im)

def evaluate(im : np.ndarray):
	"""Evaluate an image.

	Vars:
		im: the image to evaluate.

	Returns:
		# rewards similar values of color quantity. The closer to equal, the higher the score.
		# returns an integer representing sum difference from the average.
	"""
	im2 = im.reshape(-1, 3)
	uniquities = np.unique(im2, axis=0, return_counts=True)[1]
	holster = im2.shape[0] / uniquities.shape[0]
	difference = np.sum(abs(uniquities - holster))
	return (difference)

def main():
	parser = ArgumentParser(
    	prog='painter',
    	description='creates paintings according to a genetic algorithm'
	)


	parser.add_argument('-g', '--generations', default=100, help="The number of generations to run", type=int)
	parser.add_argument('-p', '--pools', default=10, help="The size of the pool", type=int)
	parser.add_argument('-m', '--mutation', default=.2, help="The chance of a mutation", type=float)
	parser.add_argument('-r', '--recombine', default = 5, help="The number of pairs to recombine in each generation", type=int)
	args = parser.parse_args()

	red = np.zeros((y,x,3))
	red[:,:,0] = 1

	blue = np.zeros((y, x, 3))
	blue[:,:,2] = 1

	states = []

	while len(states) < args.pools:
		states.append(recombine(red, blue))
	
	# creates an array with evaluate x for every element in states
	values = np.array([evaluate(z) for z in states])
	# converts values into the indices of the highest scoring elements in states.
	values = np.argsort(values)

	for i in range(0, args.generations):
		# checks if it needs to mutate, then mutates, every element in states.
		for j in range(0, len(values)):
			if (random.random() <= args.mutation):
				states[j] = mutate(states[j])
		# states is still sorted by evaluate. Changing 1 color doesn't affect evaluation, so it doesn't need to be rechecked.
		phVals = []
		# iterates backwards from the "surviving" states.
		for k in range(0, args.pools):
			phState = states[args.recombine - 1]
			for j in range(args.recombine - 2, -1, -1):
				if (random.random() < .5):
					phState = recombine(states[j], phState)
				else:
					phState = recombine(phState, states[j])
			phVals.append(phState)
		# states should now theoretically be a recombined list. I just need to sort it.
		states = phVals
		# same as above.
		values = np.array([evaluate(z) for z in states])
		values = np.argsort(values)


	# uncomment the lines below to view the image
	print(evaluate(states[0]))
	plt.imsave("art3.tiff", states[0])
	plt.show() 

	
if __name__ == '__main__':
	main()

