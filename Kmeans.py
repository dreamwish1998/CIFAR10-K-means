import numpy as np
import copy
from tqdm import tqdm  # tqdm function
from scipy.spatial.distance import pdist, euclidean

class KMeans:

    def __init__(self, n_clusters=3, max_iter=500):
        self.n_clusters = n_clusters  # cluater total numbers
        self.max_iter = max_iter  # max value of iterations
        self.loss_per_iteration = []

    def init_centroids(self):
        np.random.seed(np.random.randint(0, 100000))  # random init
        self.centroids = []
        for i in range(self.n_clusters):
            rand_index = np.random.choice(range(len(self.fit_data)))  # k-center choices
            self.centroids.append(self.fit_data[rand_index])

    def init_clusters(self):
        self.clusters = {'data': {i: [] for i in range(self.n_clusters)}}  # set data
        self.clusters['labels'] = {i: [] for i in range(self.n_clusters)}  # set labels

    def fit(self, fit_data, fit_labels):
        self.fit_data = fit_data  # data
        self.fit_labels = fit_labels  # labels
        self.predicted_labels = [None for _ in range(self.fit_data.shape[0])]  # the predictions
        self.init_centroids()  # call the function
        self.iterations = 0
        old_centroids = [np.zeros(shape=(fit_data.shape[1],)) for _ in range(self.n_clusters)]
        while not self.converged(self.iterations, old_centroids, self.centroids):
            old_centroids = copy.deepcopy(self.centroids)  # get the center before this iteration
            self.init_clusters()
            for j, sample in tqdm(enumerate(self.fit_data)):  # loop
                min_dist = float('inf')  # init min distance
                for i, centroid in enumerate(self.centroids):
                    dist = np.linalg.norm(sample - centroid)  # l2 distance
                    if dist < min_dist:
                        min_dist = dist  # upgrade rule
                        self.predicted_labels[j] = i  # set label
                if self.predicted_labels[j] is not None:
                    self.clusters['data'][self.predicted_labels[j]].append(sample)  # set data
                    self.clusters['labels'][self.predicted_labels[j]].append(self.fit_labels[j])  # set label
            self.reshape_cluster()  # call this function
            self.update_centroids()  # call this function
            self.calculate_loss()  # call the loss function (unsupervised)
            # print("\nIteration:", self.iterations, 'Loss:', self.loss, 'Difference:', self.centroids_dist)
            print("\nIteration:", self.iterations)
            self.iterations += 1
        self.calculate_accuracy()

    def update_centroids(self):
        for i in range(self.n_clusters):
            cluster = self.clusters['data'][i]
            if cluster == []:
                self.centroids[i] = self.fit_data[np.random.choice(range(len(self.fit_data)))]  # new center
            else:
                self.centroids[i] = np.mean(np.vstack((self.centroids[i], cluster)), axis=0)  # same as above

    def reshape_cluster(self):
        for id, mat in list(self.clusters['data'].items()):
            self.clusters['data'][id] = np.array(mat)  # reshape

    def converged(self, iterations, centroids, updated_centroids):  # the rule to break the loop
        if iterations > self.max_iter:
            return True
        self.centroids_dist = np.linalg.norm(np.array(updated_centroids) - np.array(centroids))  # set the center distance
        if self.centroids_dist <= 1e-50:
            print("Converged! With distance:", self.centroids_dist)  # distance < 1e-50
            return True
        return False

    def calculate_loss(self):
        self.loss = 0
        for key, value in list(self.clusters['data'].items()):
            if value is not None:
                for v in value:
                    self.loss += np.linalg.norm(v - self.centroids[key])  # l2 loss function
        self.loss_per_iteration.append(self.loss)

    def calculate_accuracy(self):
        self.clusters_labels = []
        self.clusters_info = []
        self.clusters_accuracy = []
        for clust, labels in list(self.clusters['labels'].items()):  # calculate the total accuracy
            if isinstance(labels[0], (np.ndarray)):
                labels = [l[0] for l in labels]
            occur = 0
            max_label = max(set(labels), key=labels.count)  # get the maximum
            self.clusters_labels.append(max_label)
            for label in labels:
                if label == max_label:  # calculate the same data
                    occur += 1
            acc = occur / len(list(labels))
            self.clusters_info.append([max_label, occur, len(list(labels)), acc])  # get the cluster information
            self.clusters_accuracy.append(acc)
            self.accuracy = sum(self.clusters_accuracy) / self.n_clusters
        self.labels_ = []
        for i in range(len(self.predicted_labels)):
            self.labels_.append(self.clusters_labels[self.predicted_labels[i]])  # get the ground truth and prediction
        print('[cluster_label,no_occurence_of_label,total_samples_in_cluster,cluster_accuracy]', self.clusters_info)
        print('Accuracy:', self.accuracy)


