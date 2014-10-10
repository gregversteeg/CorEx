# Run tests with nosetests

import corex
import numpy as np
from functools import partial, update_wrapper

verbose = False
seed = 3

def generate_data(n_samples=100, group_sizes=[2], dim_hidden=2, missing=0):
    Y_true = [np.random.randint(0, dim_hidden, n_samples) for _ in group_sizes]
    X = np.hstack([np.repeat(Y_true[i][:,np.newaxis], size, axis=1) for i, size in enumerate(group_sizes)])
    clusters = [i for i in range(len(group_sizes)) for _ in range(group_sizes[i])]
    tcs = map(lambda z: (z-1)*np.log(dim_hidden), group_sizes)
    X = np.where(np.random.random(X.shape) >= missing, X, -1)
    return X, Y_true, clusters, tcs

def generate_noisy_data(n_samples=100, group_sizes=[2], erasure_p=0):
    # Implement an erasure channel with erasure probability erasure_p
    # The capacity of a single such channel is 1-erasure_p,
    # So if we have group_size < 1/(1-p) , Shannon's bound forbids perfect recovery
    # Or, 1 - 1/g <  p
    dim_hidden = 3
    Y_true = [np.random.randint(0, 2, n_samples) for _ in group_sizes]
    X = np.hstack([np.repeat(Y_true[i][:,np.newaxis], size, axis=1) for i, size in enumerate(group_sizes)])
    X = np.where(np.random.random(X.shape) < erasure_p, 2, X)  # Erasure channel
    clusters = [i for i in range(len(group_sizes)) for _ in range(group_sizes[i])]
    tcs = map(lambda z: (z-1)*np.log(2), group_sizes)
    return X, Y_true, clusters, tcs

def check_correct(clusters, tcs, Y_true, X, corex):
    assert np.array_equal(corex.transform(X), corex.labels)  # Correctness of transform
    assert np.array_equal(corex.clusters, clusters), str(zip(corex.clusters, clusters))  # Check connections
    for j, tc in enumerate(tcs):
        assert np.abs(corex.tcs[j]-tc)/tc < 0.1, "Values %f, %f" %(corex.tcs[j], tc)  # TC relative error is small
        assert len(set(map(tuple, zip(corex.labels.T[j], Y_true[j])))) == len(set(Y_true[j])), \
          zip(corex.labels.T[j], Y_true[j])  # One-to-one correspondence of labels

def test_corex_all():
    n_samples = 100
    for group_sizes in [[2], [3, 2]]:
        for dim_hidden in [2, 3]:
            np.random.seed(seed)
            X, Y_true, clusters, tcs = generate_data(n_samples=n_samples, group_sizes=group_sizes, dim_hidden=dim_hidden)
            methods = [
                corex.Corex(n_hidden=len(group_sizes), dim_hidden=dim_hidden, missing_values=-1, seed=seed, verbose=verbose).fit(X)
                ]
            for i, method in enumerate(methods):
                f = partial(check_correct, clusters, method.tcs, Y_true, X, method)
                update_wrapper(f, check_correct)
                f.description = 'method: ' + ['base', 'gaussian', 'discrete', 'discrete NT', 'gaussian NT', 'beta NT'][i] + \
                                ', groups:' + str(group_sizes) + ', dim_hidden:' + str(dim_hidden) + ', seed: '+str(seed)
                yield (f, )

def test_missing_values():
    n_samples = 100
    dim_hidden = 2
    missing = 0.1
    group_sizes = [10, 7]  # Chance of entire row missing smaller than missing^n
    np.random.seed(seed)
    X, Y_true, clusters, tcs = generate_data(n_samples=n_samples, group_sizes=group_sizes,
                                                     dim_hidden=dim_hidden, missing=missing)
    methods = [
        corex.Corex(n_hidden=len(group_sizes), dim_hidden=dim_hidden, missing_values=-1, seed=seed, verbose=verbose).fit(X)
    ]

    for i, method in enumerate(methods):
        f = partial(check_correct, clusters, method.tcs, Y_true, X, method)
        update_wrapper(f, check_correct)
        f.description = 'missing values, '+ ['base', 'gaussian', 'discrete', 'discrete NT', 'gaussian NT'][i] + ', seed: '+str(seed)
        yield (f, )

def test_near_shannon_limit():
    X, Y_true, clusters, tcs = generate_noisy_data(n_samples=1000, group_sizes=[200], erasure_p=1.-3./200)
    out = corex.Corex(n_hidden=1, seed=seed, verbose=verbose).fit(X)
    assert max(np.mean(Y_true==out.labels.T), 1-np.mean(Y_true==out.labels.T)) > 0.95  # rate = 3*capacity, near perfect

    X, Y_true, clusters, tcs = generate_noisy_data(n_samples=1000, group_sizes=[200], erasure_p=1.-1./200)
    out = corex.Corex(n_hidden=1, seed=seed, verbose=verbose).fit(X)
    assert max(np.mean(Y_true==out.labels.T), 1-np.mean(Y_true==out.labels.T)) < 0.9  # rate=capacity, not perfect