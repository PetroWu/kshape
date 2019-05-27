# from kshape_local.core import kshape, zscore
import sys
sys.path.append('.')
import numpy as np

'''
kshape demos
'''
# from kshape_local.core import kshape, zscore, kshape_infer, kmshape, kmshape_infer

# time_series = [[1,2,3,4,5], [0,1,2,3,4], [3,2,1,0,-1], [1,2,2,3,3]]
# cluster_num = 2
# print(zscore(time_series))
# clusters = kshape(zscore(time_series), cluster_num, step=1000)
# print(clusters)
# centroids = np.array([c[0] for c in clusters])
# test_series = [[1,2,3,4,5], [0,1,2,3,4], [3,2,1,0,-1], [1,2,2,3,3]]
# print(kshape_infer(test_series, centroids))

# time_series = [[1,2,3,4,5], [0,1,2,3,4], [3,2,1,0,-1], [1,2,2,3,3]]
# cluster_num = 2
# L = 2
# print(zscore(time_series))
# idx, centroids = kmshape(zscore(time_series), cluster_num, L, step=1000)
# print('train', idx)
# test_series = [[1,2,3,4,5], [0,1,2,3,4], [3,2,1,0,-1], [1,2,2,3,3]]
# print('infer', kmshape_infer(test_series, L, centroids))


'''
kline demos
'''
from kshape_local.knnca import knnca
kline_series = [
        [[1000, 1005, 1010, 996], [1003, 1002, 1013, 1000], [1003, 1008, 1008, 1003]],
        [[1001, 1006, 1011, 997], [1004, 1002, 1014, 1001], [1004, 1009, 1004, 1009]],
        [[1009, 1001, 1011, 997], [1009, 1001, 1009, 1001], [1014, 999, 1024, 996]],
        [[1008, 1000, 1010, 996], [1008, 1000, 1008, 1000], [1013, 998, 1023, 995]],
    ]
print(knnca(kline_series))