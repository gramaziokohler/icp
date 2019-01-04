import numpy as np
import time
import icp

import logging, sys

# Constants
N = 1000                                      # number of random points in the dataset
num_tests = 1                               # number of test iterations
dim = 3                                     # number of dimensions of the points
noise_sigma = .01                           # standard deviation error to be added
translation = .5                            # max translation of the test set
rotation = .1                               # max rotation (radians) of the test set


def test_best_fit():

    # Generate a random dataset
    A = np.random.rand(N, dim)

    total_time = 0

    for i in range(num_tests):

        B = np.copy(A)

        # Translate
        t = np.random.rand(dim)*translation
        B += t

        # Rotate
        R = icp.rotation_matrix(np.random.rand(dim), np.random.rand()*rotation)
        B = np.dot(R, B.T).T

        # Add noise
        B += np.random.randn(N, dim) * noise_sigma

        # Find best fit transform
        start = time.time()
        T, R1, t1 = icp.best_fit_transform(B, A)
        total_time += time.time() - start

        # Make C a homogeneous representation of B
        C = np.ones((N, 4))
        C[:,0:3] = B

        # Transform C
        C = np.dot(T, C.T).T

        assert np.allclose(C[:,0:3], A, atol=6*noise_sigma) # T should transform B (or C) to A
        assert np.allclose(-t1, t, atol=6*noise_sigma)      # t and t1 should be inverses
        assert np.allclose(R1.T, R, atol=6*noise_sigma)     # R and R1 should be inverses

    print('best fit time: {:.3}'.format(total_time/num_tests))

    return


def test_icp_default():

    # Generate a random dataset
    A = np.random.rand(N, dim)

    total_time = 0

    for i in range(num_tests):

        B = np.copy(A)

        # Translate
        t = np.random.rand(dim)*translation
        B += t

        # Rotate
        R = icp.rotation_matrix(np.random.rand(dim), np.random.rand() * rotation)
        B = np.dot(R, B.T).T

        # Add noise
        B += np.random.randn(N, dim) * noise_sigma

        # Shuffle to disrupt correspondence
        np.random.shuffle(B)

        # Run ICP
        start = time.time()
        T, distances, iterations = icp.icp(B, A, convergence=0.000001)
        total_time += time.time() - start

        # Make C a homogeneous representation of B
        C = np.ones((N, 4))
        C[:,0:3] = np.copy(B)

        # Transform C
        C = np.dot(T, C.T).T

        assert np.mean(distances) < 6*noise_sigma                   # mean error should be small
        assert np.allclose(T[0:3,0:3].T, R, atol=6*noise_sigma)     # T and R should be inverses
        assert np.allclose(-T[0:3,3], t, atol=6*noise_sigma)        # T and t should be inverses

    print('icp time: {:.3}'.format(total_time/num_tests))

    return


def test_icp():

    # Load Ply A and B
    A = icp.loadPointCloud('PtCloud1.ply')
    logging.debug("Point Cloud A: %d pts" % len(A))
    B = icp.loadPointCloud('PtCloud2.ply')
    logging.debug("Point Cloud B: %d pts" % len(B))

    #Create a low density point cloud for first pass
    B_low = icp.decimate_by_sequence(B,20)
    logging.debug("Point Cloud B: %d pts (After decimate(16))" % len(B_low))

    #Create a high density but centrally filtered point cloud for senond pass
    B = icp.filter_points_by_angle(B,20) #Filter point clouds within 25degrees in Z Axis
    logging.debug("Point Cloud B : %d pts (After filtering)" % len(B))

    total_time = 0

    # Run ICP
    start = time.time()
    T, distances, iterations = icp.icp(B_low, A, convergence=0.00001, standard_deviation_range = 0.0, max_iterations=30)
    T, distances, iterations = icp.icp(B, A, convergence=0.0000001, standard_deviation_range = 0.0, max_iterations=100, init_pose = T , quickconverge = 2)

    total_time += time.time() - start


    print('icp time: {:.3}'.format(total_time))

    return


if __name__ == "__main__":
    test_icp()
