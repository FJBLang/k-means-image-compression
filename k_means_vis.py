import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio

# Image file to compress
img_file = "example_img.jpg"
out_file = "example_output.gif"

# Number of clusters
K = 16
# Maximum number of iternations
MAX_ITER = 20

# Styling
TITLE_SIZE = 18
AXIS_LABEL_SIZE = 12
AXIS_TICK_SIZE = 8
PIXEL_SCATTER_SIZE = 1
CENTROID_SIZE = 16	
FRAME_DURATION = 0.6

def decompress_img(idx, centroids, img_shape):
	"""use idx and centroids to compute an array of pixels that can be
	used by matplotlib to display the compressed image"""
	centroids = np.rint(centroids)
	centroids = centroids.astype(int)
	img = centroids[idx].reshape(img_shape)
	return img

def vis_iteration(X, idx, centroids, img_shape, i):
	"""plot a visualization of the current iteration and save as a 
	jpeg file"""
	img = decompress_img(idx, centroids, img_shape)
	col_array = c=np.round(centroids[idx]/255.0,4)
	
	fig = plt.figure(figsize=(10,8), dpi=100)
	fig.suptitle("Iteration: " + str(i), fontsize=TITLE_SIZE)
	
	clrs = ["R","G","B"]
	for n, c in enumerate([["G","R"],["B","R"],["G","B"]]):
		plt.subplot(2,2,n+1)
		plt.scatter(X[:,clrs.index(c[0])], X[:,clrs.index(c[1])], c=col_array, s=PIXEL_SCATTER_SIZE)
		plt.scatter(centroids[:,clrs.index(c[0])], centroids[:,clrs.index(c[1])], c='red', s=CENTROID_SIZE)
		plt.axis([0,255,0,255])
		plt.xlabel(c[0], fontsize=AXIS_LABEL_SIZE)
		plt.ylabel(c[1], fontsize=AXIS_LABEL_SIZE)
		plt.tick_params(axis='both', labelsize=AXIS_TICK_SIZE)

	plt.subplot(2,2,4)
	plt.axis('off')
	plt.imshow(img)
	
	plt.savefig("iteration_%02d.png" %i, bbox_inches='tight')
	plt.close()
	
def vis_original(X, img_shape):
	"""plot of visualization of the original image"""
	img = X.reshape(img_shape)
	col_array = c=np.round(X/255.0,4)
	
	fig = plt.figure(figsize=(10,8), dpi=100)
	fig.suptitle("Original Image", fontsize=TITLE_SIZE)
		
	clrs = ["R","G","B"]
	for n, c in enumerate([["G","R"],["B","R"],["G","B"]]):
		plt.subplot(2,2,n+1)
		plt.scatter(X[:,clrs.index(c[0])], X[:,clrs.index(c[1])], c=col_array, s=PIXEL_SCATTER_SIZE)
		plt.axis([0,255,0,255])
		plt.xlabel(c[0], fontsize=AXIS_LABEL_SIZE)
		plt.ylabel(c[1], fontsize=AXIS_LABEL_SIZE)
		plt.tick_params(axis='both', labelsize=AXIS_TICK_SIZE)
	
	plt.subplot(2,2,4)
	plt.axis('off')
	plt.imshow(img)
	
	plt.savefig("iteration_00.png", bbox_inches='tight')
	plt.close()
	
def init_centroids(X, K):
	"""initialize the centroids to be a sample of K points in X
	(sampling without replacement)"""
	Xlen = np.shape(X)[0]
	centroids = X[random.sample(range(Xlen),K)]
	return centroids.astype(float)
	
def find_closest_centroids(X, centroids):
	"""return an array 'idx', the i'th element of which is the index of
	the centroid closest to the i'th element of X"""
	Xlen = np.shape(X)[0]
	# for each pixel in X find the closest centroid
	idx = np.zeros(Xlen, dtype=np.int8)
	for xidx in range(Xlen):
		abs_dif = np.sum(np.abs(centroids - X[xidx])**2, axis=-1)
		idx[xidx] = np.argmin(abs_dif)
	return idx
	
def update_centroids(X, idx):
	"""Update the centroids so that each ith centroid is the average 
	point of all points in X assigned the it"""
	centroids = np.zeros((K,3))
	for i in range(K):
		centroids[i] = np.mean(X[idx == i], axis=0)
	return centroids
	
def k_means_vis(img_file):

	# Load the image file
	img = mpimg.imread(img_file)
	img_shape = img.shape
	
	# Reshape image array to a list of pixels
	X = img.reshape(-1,3)
	
	# Visualise original images colour distribution
	vis_original(X, img_shape)
	
	# Apply the k-means algorithm and save a visualization of each iteration
	i = 1
	converged = False
	centroids = init_centroids(X, K)
	while (converged == False and i <= MAX_ITER):
		idx = find_closest_centroids(X, centroids)
		vis_iteration(X, idx, centroids, img_shape, i)
		
		old_centroids = centroids
		centroids = update_centroids(X, idx)
		if (old_centroids == centroids).all():
			converged = True
			
		i = i + 1
	
	# Turn the jpgs into a gif and delete the jpgs
	images = []
	for j in range(i):
		img_file = "iteration_%02d.png" %j
		images.append(imageio.imread(img_file))
		# Double the duration of the first and last frame
		if j == 0 or j == i-1:
			images.append(imageio.imread(img_file))
		os.remove(img_file)
	imageio.mimsave(out_file, images, duration=FRAME_DURATION)
	
k_means_vis(img_file)
