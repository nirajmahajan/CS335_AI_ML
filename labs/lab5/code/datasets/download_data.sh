echo "Downloading CIFAR-10"
wget "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
tar -xzvf cifar-10-python.tar.gz
mv cifar-10-batches-py cifar-10

echo "Downloading MNIST"
wget "https://www.cse.iitb.ac.in/~satvikmashkaria/mnist.pkl.gz"
