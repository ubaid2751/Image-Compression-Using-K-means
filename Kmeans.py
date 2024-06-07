from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class K_means():
    def __init__(self):
        pass
    
    def kmeans_init_centroids(self, X, k):
        randidx = np.random.permutation(X.shape[0])
        centroids = X[randidx[:k]]
        
        return centroids
    
    def compute_centroids(self, X, idx, K):
        m, n = X.shape
        
        centroids = np.zeros((K, n))
        
        for k in range(K):
            points = X[idx == k]
            if points.size == 0:
                # Avoid empty cluster by reinitializing the centroid randomly
                centroids[k] = X[np.random.choice(X.shape[0])]
            else:
                centroids[k] = np.mean(points, axis=0)
            
        return centroids
    
    def find_closest_centroids(self, X, centroids):
        k = centroids.shape[0]
        idx = np.zeros(X.shape[0], dtype=int)
        
        for i in range(X.shape[0]):
            distance = []
            for j in range(centroids.shape[0]):
                norm_ij = np.linalg.norm(X[i] - centroids[j])
                distance.append(norm_ij)
            
            idx[i] = np.argmin(distance)
            
        return idx
    
    @staticmethod
    def draw_line(p1, p2, style="-k", linewidth=1):
       plt.plot([p1[0], p2[0]], [p1[1], p2[1]], style, linewidth=linewidth)

    def plot_data_points(self, X, idx, K):
        cmap = ListedColormap(plt.cm.get_cmap('tab10').colors[:K])
        plt.scatter(X[:, 0], X[:, 1], c=idx, cmap=cmap, edgecolor='k', s=20, alpha=0.7)


    def plot_kmeans(self, X, centroids, previous_centroids, idx, K, i):
        self.plot_data_points(X, idx, K)
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', linewidths=3, c='black')
        
        for j in range(centroids.shape[0]):
            self.draw_line(centroids[j, :], previous_centroids[j, :])
            
        plt.title("Iteration number %d" %i)        

    def run_kmeans(self, X, K, progress_bar, centroids, max_iters=10, plot_progress=False):
        m, n = X.shape
        previous_centroids = centroids
        idx = np.zeros(m)

        for i in range(max_iters):
            # print("K-Means iteration %d/%d" % (i, max_iters-1))
            progress_bar.progress(i * 10)
            idx = self.find_closest_centroids(X, centroids)
            
            if plot_progress:
                self.plot_kmeans(X, centroids, previous_centroids, idx, K, i)
            
            previous_centroids = centroids    
            centroids = self.compute_centroids(X, idx, K)
            
        return centroids, idx
    
def normalize(df):
    return (df - np.mean(df, axis=0)) / np.std(df, axis=0)

def show_centroid_colors(centroids):
    palette = np.expand_dims(centroids, axis=0)
    num = np.arange(0,len(centroids))
    plt.figure(figsize=(16, 16))
    plt.xticks(num)
    plt.yticks([])
    plt.imshow(palette)
    plt.show()