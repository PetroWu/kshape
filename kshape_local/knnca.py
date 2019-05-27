# coding=utf-8
# implemented from <<K-Line Patterns’ Predictive Power Analysis Using the Methods of Similarity Match and Clustering>>
import numpy as np


def upper_shadow(candle):
    '''
        candle: [open, close, high, low]
        return the length of upper shadow
    '''
    assert candle[1] > 0
    if candle[0] >= candle[1]:
        return (candle[2] - candle[0]) * 10. / candle[1]
    return (candle[2] - candle[1]) * 10. / candle[1]

def similarity_upper_shadow(candle_i, candle_j):
    usi = upper_shadow(candle_i)
    usj = upper_shadow(candle_j)
    if usi > 0 or usj > 0:
        return min(usi, usj) * 1. / max(usi, usj)
    return 1.

def lower_shadow(candle):
    assert candle[1] > 0
    if candle[0] >= candle[1]:
        return (candle[1] - candle[3]) * 10. / candle[1]
    return (candle[0] - candle[3]) * 10. / candle[1]

def similarity_lower_shadow(candle_i, candle_j):
    lsi = lower_shadow(candle_i)
    lsj = lower_shadow(candle_j)
    if lsi > 0 or lsj > 0:
        return min(lsi, lsj) * 1. / max(lsi, lsj)
    return 1.

def body_length(candle):
    assert candle[1] > 0
    return (candle[1] - candle[0]) * 10. / candle[1]

def similarity_body(candle_i, candle_j):
    bi = body_length(candle_i)
    bj = body_length(candle_j)
    bij = bi * bj
    abi = abs(bi)
    abj = abs(bj)
    if bij > 0:
        return min(abi, abj) / max(abi, abj)
    elif bij < 0:
        return - min(abi, abj) / max(abi, abj)
    elif abi != abj:
        return 0.
    return 1.

def shape_similarity_series_pair(series_i, series_j, weights):
    '''
        series_i, series_j: list of candles, numpy matrix!!!
        weights: [w_body, w_upper_shadow, w_lower_shadow]
    '''
    leni = series_i.shape[0]
    lenj = series_j.shape[0]
    assert leni == lenj
    return sum([weights[0] * similarity_body(ci, cj) + weights[1] * similarity_upper_shadow(ci, cj) + weights[2] * similarity_lower_shadow(ci, cj)  for ci, cj in zip(series_i, series_j)]) / leni

def relative_position(candle_i, candle_front):
    '''
        candle_front: C_{t-1}, is None if i is 0
        according to Eq.12, return y_t
    '''
    if candle_front is None: # the first candle
        return 1.
    return (candle_i[1] - candle_front[1]) * 10. / candle_front[1]

def position_similarity(candle_i, candle_fi, candle_j, candle_fj):
    yi = relative_position(candle_i, candle_fi)
    yj = relative_position(candle_j, candle_fj)
    ayi = abs(yi)
    ayj = abs(yj)
    yij = yi * yj
    if yij > 0:
        return min(ayi, ayj) / max(ayi, ayj)
    elif yij < 0:
        return - min(ayi, ayj) / max(ayi, ayj)
    elif ayi != ayj:
        return 0.
    return 1.

def position_similarity_series_pair(series_i, series_j):
    '''
        series_i, series_j: list of candles, python list!!!
    '''
    leni = len(series_i)
    lenj = len(series_j)
    assert leni == lenj
    psi = [None] + series_i[:-1]
    psj = [None] + series_j[:-1]
    return sum([position_similarity(ci, fi, cj, fj) for ci, fi, cj, fj in zip(series_i, psi, series_j, psj)]) / leni

def kline_similarity(series_i, series_j, weights):
    '''
        series_i, series_j: list of candles, python list!!!
        weights: [ws, wp, w_body, w_upper_shadow, w_lower_shadow]
    '''
    return shape_similarity_series_pair(np.array(series_i), np.array(series_j), weights[2:]) * weights[0] + position_similarity_series_pair(series_i, series_j) * weights[1]

# TODO: too naive, can be applied into soft-DTW
def knnca(series, weights=None, iter_steps=200, similarity_threshold=.5):
    '''
        series: list of series, python list!!!
        weights: [ws, wp, w_body, w_upper_shadow, w_lower_shadow]
        no need to set number of clusters
    '''
    lens = len(series)
    assert lens > 0
    if weights is None:
        weights = [.2, .8, .6, .2, .2]
    C_set, m = [[0]], 0

    for i in range(1, lens):
        sim_max, max_f = 0, 0
        for item, Q_item in enumerate(C_set):
            for sj in Q_item:
                tmp_sim = kline_similarity(series[i], series[sj], weights)
                if tmp_sim > sim_max:
                    sim_max = tmp_sim
                    max_f = item
        # print(sim_max)
        if sim_max > similarity_threshold:
            C_set[max_f].append(i)
        else:
            C_set.append([i]) # add a new cluster
            m += 1
    return C_set
    
# def kline_means(series, weights=None, num_clusters=2, iter_steps=200):
#     if weights is None:
#         weights = [.2, .8, .6, .2, .2]
#     num_series = len(series)
#     series_matrix = np.array(series)
#     idx = np.arange(num_series)
#     np.random.shuffle(idx)
#     centroids = idx[:num_clusters]  # randomly pick #num_clusters points as initial centroids
#     for iter_cnt in range(iter_steps):
#         sims = np.zeros((num_clusters, num_series))
#         for i in range(num_clusters):
#             sims[i] = np.array([kline_similarity(si, series[centroids[i]], weights) for si in series])
#         tmp_clusters = np.argmax(sims, axis=0)
#         # calculate new centroids
#         for i in range(num_clusters):
            