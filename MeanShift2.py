import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot as plt
from matplotlib import style
style.use('ggplot')
from mpl_toolkits.mplot3d import Axes3D


#data = np.load('National_data2.csv', encoding='latin1')
X = np.load('National_data2.csv', encoding='latin1')

print (X)