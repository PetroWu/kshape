# from kshape_local.core import kshape, zscore
import sys
sys.path.append('.')
import numpy as np
from kshape_local.core import kshape, zscore, kshape_infer, kmshape, kmshape_infer

# time_series = [[1,2,3,4,5], [0,1,2,3,4], [3,2,1,0,-1], [1,2,2,3,3]]
# cluster_num = 2
# print(zscore(time_series))
# clusters = kshape(zscore(time_series), cluster_num, step=1000)
# print(clusters)
# centroids = np.array([c[0] for c in clusters])
# test_series = [[1,2,3,4,5], [0,1,2,3,4], [3,2,1,0,-1], [1,2,2,3,3]]
# print(kshape_infer(test_series, centroids))

time_series = [[1,2,3,4,5], [0,1,2,3,4], [3,2,1,0,-1], [1,2,2,3,3]]
cluster_num = 2
L = 2
print(zscore(time_series))
idx, centroids = kmshape(zscore(time_series), cluster_num, L, step=1000)
print('train', idx)
test_series = [[1,2,3,4,5], [0,1,2,3,4], [3,2,1,0,-1], [1,2,2,3,3]]
print('infer', kmshape_infer(test_series, L, centroids))