import pickle
import numpy as np
from os.path import join
from os import listdir
import matplotlib.pyplot as plt
from tqdm import tqdm
import struct as st


class DataReader:

    def __init__(self, root_dir, type='cifar-10'):
        self.root_dir = root_dir  # the root data
        self.type = type  # process data

    def get_dict_from_pickle(self):
        self.train_dict = unpickle(join(self.root_dir, 'train'))  # train data
        self.test_dict = unpickle(join(self.root_dir, 'test'))  # test data

    def get_train_data(self):
        if self.type == 'cifar-100':  # big data
            self.get_dict_from_pickle()
            # data = np.array(self.train_dict[b'data'])
            # lbls_sub = np.array(self.train_dict[b'fine_labels'])
            # lbls_class = np.array(self.train_dict[b'coarse_labels'])
            # return data, lbls_class, lbls_sub
        elif self.type == 'cifar-10':  # cifar-10
            data = []
            labels = []
            print("Loading data now")
            for file_ in tqdm(listdir(self.root_dir)):
                if file_.split('_')[0] == 'data':
                    dict = unpickle(join(self.root_dir, file_))  # unpickle the data
                    data.extend(dict[b'data'])  # data part
                    labels.extend(dict[b'labels'])  # label part
            return np.array(data), np.array(labels), None


    def get_test_data(self):  # same as above
        if self.type == 'cifar-100':
            self.get_dict_from_pickle()
            # data = np.array(self.test_dict[b'data'])
            # lbls_sub = np.array(self.test_dict[b'fine_labels'])
            # lbls_class = np.array(self.test_dict[b'coarse_labels'])
            # return data, lbls_class, lbls_sub
        elif self.type == 'cifar-10':
            data = np.empty(shape=(0, 3072))
            # labels = []
            # for file_ in listdir(self.root_dir):
            #     if file_.split('_')[0] == 'test':
            #         dict = unpickle(join(self.root_dir, file_))
            #         data = np.vstack((data, dict[b'data']))
            #         print(data[data.shape[0] - 1])
            #         labels.append(dict[b'labels'])
            # return np.array(data), np.array(labels), None

    def reshape_to_plot(self, data):

        return data.reshape(data.shape[0], 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")  # reshape data

    def plot_imgs(self, in_data, in_data_label, t, random=False):
        data = np.array([d for d in in_data])  # from train data
        data = self.reshape_to_plot(data)  # reshape data
        class_idx = np.arange(t).tolist()  # total k num
        select_idx = []
        for n in range(t):  # show the selected images
            co = 0
            for m in range(len(in_data_label)):  # get related class data
                if in_data_label[m] == class_idx[n] and co < 5:
                    select_idx.append(m)
                    co = co + 1
        x = t
        y = 5
        fig, ax = plt.subplots(x, y, figsize=(5, 5))  # sub-images
        i = 0
        for j in range(x):
            for k in range(y):
                # if random:
                #     i = np.random.choice(range(len(data)))
                ax[j][k].set_axis_off()  # set the sub-images
                if k == 2:
                    ax[j][k].set_title("5 examples of Cluster {0}".format(j))
                ax[j][k].imshow(data[select_idx[i]:select_idx[i]+1][0])  # show images
                i += 1
        plt.xlabel("Example")
        plt.show()

    def plot_img(self, data):
        if self.type != 'mnist':
            assert data.shape == (3072,)
            data = data.reshape(1, 3072)  # reshape
            data = data.reshape(data.shape[0], 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")  # reshape
        # elif self.type == 'mnist':
        #     assert data.shape == (28 * 28,)
        #     data = data.reshape(1, 28, 28).astype('uint8')
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(data[0])
        plt.show()

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')  # unpickle data
    return dict