from sklearn import datasets
import numpy as np

def isNotebook(): 
    # This function would return True if it's run inside a notebook or ipython environment. 
    # False if it's in normal Python interpreter. Otherwise, it'll raise an Error.
    # It is not necessary for you to understand this function.
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell' or shell == 'TerminalInteractiveShell' or shell == 'Shell':
            return True
        else:
            raise NotImplementedError
    except NameError:
        return False

def plot_diagram(X, y=None, k=None):
    if not isNotebook():
        return
def plot_points(points, marker='o'):
    if not isNotebook():
        return

if isNotebook():
    secret_number = 42 # This will be the seed of the random number generator
else:
    import pickle
    with open('secret_number.pkl', 'rb') as f:
        secret_number = pickle.load(f) # Only matters during grading

X, y = datasets.make_blobs(10000, 2, centers=3, random_state=secret_number)  # Create a dataset with 3 blobs of cluster

    
    
class KCluster:
    def __init__(self, k, X, ndim=2):
        self.k = k
        self.ndim = ndim
        # Choose k data points from X as the initial centroids.
        self.centroid = X[np.random.randint(0, len(X), size=(k, ))]
    
    def run(self, X):
        # TODO: 1. Find the difference between each data point and centroid, assign the result to diff 
        # (Hints: the shape of diff should be (self.k, X.shape[0], self.ndim))
        # diff = np.zeros((self.k, X.shape[0], self.ndim))
        # for i in range(self.k):
        #     diff[i] = X - self.centroid[i]
        # TODO: 2. Calculate the Euclidean distance between each data point and centroid, assign the result to dist.
        # (Hints: Euclidean distance = ((x2 - x1)**2 + (y_2 - y_1)**2) ** 0.5. You can also check the documentation of numpy.linalg.norm)
        # TODO (optional): You can also calculate the distance between each data point and centroid directly without following the instruction above:
        # diff = np.zeros((self.k, X.shape[0], self.ndim))
        # for i in range(self.k):
        #     diff[i] = X - self.centroid[i]
            
        dist = np.zeros((self.k, X.shape[0]))
        for i in range(self.k):
            dist[i] = np.sqrt(np.sum((X - self.centroid[i])**2, axis=1))
            # if(j == 5): print(dist[i,:j])
        # TODO: 3. Assign the index of the closest centroid to each data point.
        # (Hints: use numpy.argmin to find the index of the closest centroid for each data point)
        
        # for i, x in enumerate(X):
        #     output[i] = np.argmin(dist[:, i])
        
        output = np.argmin(dist, axis=0)
            
        # TODO: 4. Update each centroid using the mean of each cluster.
        for i in range(self.k):
            for n in range(self.ndim):
                '''!!!Major Bug!!!!'''
                self.centroid[i, n] = np.mean(X[output == i, n])
        
        # print(dist[:, 0])
        return output
    
def SSE(X, y, k, centroids):
    sse = 0
    # TODO: For each cluster, calculate distance (square of difference, i.e. Euclidean/L2-distance) of samples to the datapoints and accumulate the sum to `sse`. (Hints: use numpy.sum and for loop)
    for j in range(k):
        sum_dist = np.sum((X[y == j] - centroids[j])**2)
        sse += sum_dist
    
    return sse

np.random.seed(secret_number) # Set seeds to expect the same result everytime
kmean = KCluster(3, X)
initial_points = kmean.centroids.copy()
for  n in range(100):
    output = kmean.run(X)
sse = SSE(X, output, 3, kmean.centroids)
print('SSE: ', round(sse, 13))

plot_diagram(X, output, 3)
plot_points(kmean.centroids.T, marker='+')
plot_points(initial_points.T, marker='o')

np.random.seed(2) # Set seeds to expect the same result everytime
kmean = KCluster(3, X)
initial_points = kmean.centroids.copy()
for  n in range(500):
    output = kmean.run(X)


plot_diagram(X, output, 3)
plot_points(kmean.centroids.T, marker='+')
plot_points(initial_points.T, marker='o')
SSE(X, output, 3, kmean.centroids) 

from sklearn import cluster
X, y = datasets.make_moons(10000, noise=0.05)
kmean = KCluster(2, X)
for  n in range(100):
    output = kmean.run(X)

plot_diagram(X, output, 2) # The two clusters are obviously not correctly separated.