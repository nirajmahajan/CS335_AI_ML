import argparse
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5927)

class KMeans():
	def __init__(self, D, n_clusters):
		self.n_clusters = n_clusters
		self.cluster_centers = np.zeros((n_clusters, D))

	def init_clusters(self, data):
		### TODO
		### Initialize cluster_centers using n_clusters points sampled from data
		indices = np.random.permutation(np.arange(data.shape[0]))[:self.n_clusters]
		self.cluster_centers += data[indices,:]
		### END TODO

	def pred(self, x):
		### TODO: Given a sample x, return id of closest cluster center
		# x -> dx1
		dists = ((self.cluster_centers - x.reshape(1,-1))**2).sum(1)
		return np.argmin(dists)

		### END TODO

	def train(self, data, max_iter=10000, epsilon=1e-4):
		for it in range(max_iter):
			### TODO
			### Declare and initialize required variables
			old_centres = self.cluster_centers.copy()

			### Update labels for each point

			clusters = np.argmin(((np.expand_dims(data,1) - np.expand_dims(self.cluster_centers, 0))**2).sum(2), 1)


			### Update cluster centers
			### Note: If some cluster is empty, do not update the cluster center

			for ci in range(self.n_clusters):
				filter = clusters == ci
				if filter.sum() > 0:
					self.cluster_centers[ci,:] = data[filter,:].mean(0)

			### Check for convergence
			### Stop if distance between each of the old and new cluster centers is less than epsilon

			if np.linalg.norm(self.cluster_centers - old_centres) < epsilon:
				break

			### END TODO
		return it

	def replace_by_center(self, data):
		out = np.zeros_like(data)
		for i, x in enumerate(data):
			out[i] = self.cluster_centers[self.pred(x)]
		return out

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--image', default='1', choices=['1', '2', '3'])
	parser.add_argument('--k', default=5, type=int)

	args = parser.parse_args()

	image = plt.imread(f'data/{args.image}.png')
	x = image.reshape(-1, 3)
	kmeans = KMeans(D=3, n_clusters=args.k)
	kmeans.init_clusters(x)
	kmeans.train(x)
	out = kmeans.replace_by_center(x)
	plt.imshow(out.reshape(image.shape))
	# plt.show()


	# my code 
	for image_code in range(1,4):
		fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(9, 3))
		for k_id, k in enumerate([2,5,10]):
			image = plt.imread(f'data/{image_code}.png')
			x = image.reshape(-1, 3)
			kmeans = KMeans(D=3, n_clusters=k)
			kmeans.init_clusters(x)
			kmeans.train(x)
			out = kmeans.replace_by_center(x)
			ax[k_id].imshow(out.reshape(image.shape))
			ax[k_id].set_title("Image {} | k = {}".format(image_code, k_id))
	plt.show()

