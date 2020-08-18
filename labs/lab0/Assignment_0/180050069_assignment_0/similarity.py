import argparse
import time
import numpy as np 
import matplotlib.pyplot as plt

def d(x,y):
    '''
    Given x and y where each is an np arrays of size (dim,1), compute L2 distance between them
    '''
    return (x*x).sum() + (y*y).sum() - 2*((x*y).sum())
    ## ADD CODE HERE ##
    pass


def pairwise_similarity_looped(X):
    '''
    Given X, Y where each is an np array of size (num_points_1,dim) and (num_points_2,dim), 
    return K, an array having size (num_points_1,num_points_2) according to the problem given
    '''
    ## STEP 1 - Initialize K as a numpy array - ADD CODE TO COMPUTE n1, n2 ##
    n = X.shape[0]

    K = np.zeros((n,n))

    ## STEP 2 - Loop and compute  -- COMPLETE THE LOOP BELOW ##
    
    for i in range(n):
    	for j in range(n):
    		x_vec = X[i,:]
    		y_vec = X[j,:]
    		K[i,j] = d(x_vec, y_vec)

    return K 


def pairwise_similarity_vec(X):
    '''
    Given X, Y where each is an np array of size (num_points_1,dim) and (num_points_2,dim), 
    return K, an array having size (num_points_1,num_points_2) according to the problem given

    This problem can be simplified in the following way - 
    Each entry in K has three terms (as seen in problem 2.1 (a)).
    Hence, first  computethe norm for all points, reshape it suitably,
    then compute the dot product.
    All of these can be done by using on the *, np.matmul, np.sum(), and transpose operators.
    '''
    ## ADD CODE TO COMPUTE K ##

    # x.x or y.y
    # 1,n
    component1 = (X ** 2).sum(1).reshape(1,-1)
    # n,1
    component2 = (X ** 2).sum(1).reshape(-1,1)
    component3 = X.dot(X.T)

    return (component1 + component2 - 2*(component3))

def get_time(n, d):
    X = np.random.normal(0.,1.,size=(n,d))

    t1 = time.time()
    K_loop = pairwise_similarity_looped(X)
    t2 = time.time()
    K_vec  = pairwise_similarity_vec(X)
    t3 = time.time()

    assert np.allclose(K_loop,K_vec)   # Checking that the two computations match

    return t3-t2,t2-t1

def plot_graphs():
	# Plotting for variations in dimensions
	n_fix = 10
	times_vect = []
	times_loop = []
	for di in range(100):
		d = di
		tv, tl = get_time(n_fix,d)
		times_vect.append(tv)
		times_loop.append(tl)

	fig, ax = plt.subplots()
	ax.set_title('Execution time with variations in d\n n is fixed to 10'.format(n_fix))
	ax.set_xlabel('number of dimensions')
	ax.set_ylabel('time in seconds')
	ax.plot(np.arange(len(times_vect)), np.array(times_vect), 'b', label = 'Vectorised')
	ax.plot(np.arange(len(times_loop)), np.array(times_loop), 'r', label = 'Loop')
	ax.legend()
	# plt.show()

	# Plotting for variations in n
	d_fix = 10
	times_vect = []
	times_loop = []
	for ni in range(100):
		n = ni
		tv, tl = get_time(n,d_fix)
		times_vect.append(tv)
		times_loop.append(tl)

	fig, ax = plt.subplots()
	ax.set_title('Execution time with variations in n\n d is fixed to 10'.format(n_fix))
	ax.set_xlabel('number of data points')
	ax.set_ylabel('time in seconds')
	ax.plot(np.arange(len(times_vect)), np.array(times_vect), 'b', label = 'Vectorised')
	ax.plot(np.arange(len(times_loop)), np.array(times_loop), 'r', label = 'Loop')
	ax.legend()
	plt.show()






if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--num', type=int, default=5,
                    help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=42,
                    help='Seed for random generator')
    parser.add_argument('--dim', type=int, default=2,
                    help='Lambda parameter for the distribution')


    args = parser.parse_args()

    np.random.seed(args.seed)

    X = np.random.normal(0.,1.,size=(args.num,args.dim))
    # Y = np.random.normal(1.,1.,size=(args.num,args.dim))

    t1 = time.time()
    K_loop = pairwise_similarity_looped(X)
    t2 = time.time()
    K_vec  = pairwise_similarity_vec(X)
    t3 = time.time()

    assert np.allclose(K_loop,K_vec)   # Checking that the two computations match

    np.savetxt("problem_2_loop.txt",K_loop)
    np.savetxt("problem_2_vec.txt",K_vec)
    print("Vectorized time : {}, Loop time : {}".format(t3-t2,t2-t1))
    # plot_graphs()
    