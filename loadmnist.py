from sklearn.datasets import fetch_mldata

def load_mnist():
    mnist = fetch_mldata('MNIST original')
    x,y = mnist['data'],mnist['target']
    return x,y

