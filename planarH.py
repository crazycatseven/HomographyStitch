import numpy as np
import cv2


def computeH(x1, x2):
    # Q2.2.1
    # Compute the homography between two sets of points
    # Make the matrix A
    A = np.zeros((2 * x1.shape[0], 9))

    # Move the points data into the matrix A
    for i in range(x1.shape[0]):
        x = x2[i, 0]
        y = x2[i, 1]
        x_p = x1[i, 0]
        y_p = x1[i, 1]

        A[2 * i] = [x, y, 1, 0, 0, 0, -x * x_p, -y * x_p, -x_p]
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -x * y_p, -y * y_p, -y_p]

    # Compute the SVD of A
    _, _, V = np.linalg.svd(A)

    # Store singular vector corresponding to the smallest singular value in h
    h = V[-1]
    # Reshape h into the matrix H
    H2to1 = h.reshape((3, 3))

    return H2to1


def computeH_norm(x1, x2):
    # Q2.2.2
    # Compute the centroid of the points
    c1 = np.mean(x1, axis=0)
    c2 = np.mean(x2, axis=0)

    # Shift the origin of the points to the centroid
    x1_shifted = x1 - c1
    x2_shifted = x2 - c2

    # Normalization
    s1 = np.sqrt(2) / np.std(x1_shifted)
    s2 = np.sqrt(2) / np.std(x2_shifted)

    # Make the transformation matrix T1 and T2
    T1 = np.array([[s1, 0, 0], [0, s1, 0], [0, 0, 1]]) @ np.array([[1, 0, -c1[0]], [0, 1, -c1[1]], [0, 0, 1]])
    T2 = np.array([[s2, 0, 0], [0, s2, 0], [0, 0, 1]]) @ np.array([[1, 0, -c2[0]], [0, 1, -c2[1]], [0, 0, 1]])

    # make x1 and x2 homogeneous
    x1_homogeneous = np.vstack((x1.T, np.ones(x1.shape[0])))
    x2_homogeneous = np.vstack((x2.T, np.ones(x2.shape[0])))

    # Apply the transformation to x1 and x2
    x1_norm = (T1 @ x1_homogeneous).T[:, :2]
    x2_norm = (T2 @ x2_homogeneous).T[:, :2]

    # Compute the homography between x1 and x2
    H2to1 = computeH(x1_norm, x2_norm)

    # Denormalize the homography
    H2to1 = np.linalg.inv(T1) @ H2to1 @ T2

    return H2to1


def computeH_ransac(locs1, locs2):
    # Q2.2.3
    # Compute the best fitting homography given a list of matching points

    iters = 3000
    threshold = 5
    bestH2to1 = np.zeros((3, 3))
    best_inliers = np.zeros(locs1.shape[0])

    for i in range(iters):
        # Randomly select 4 matched points from the matches list
        samples = np.random.choice(locs1.shape[0], 4, replace=False)

        x1 = locs1[samples]
        x2 = locs2[samples]

        # # Solve for the homography
        H2to1 = computeH_norm(x1, x2)

        # Score the homography, i.e, compute the inliers
        x2_homogeneous = np.vstack((locs2.T, np.ones(locs2.shape[0])))
        x2_projected = (H2to1 @ x2_homogeneous).T
        third_dim = x2_projected[:, 2:]
        third_dim[third_dim == 0] = 1
        x2_projected = x2_projected[:, :2] / third_dim

        dist = np.sqrt(np.sum((locs1 - x2_projected) ** 2, axis=1))

        # Compute inliers, 0 or 1
        inliers = dist <= threshold

        # Update
        if np.sum(inliers) > np.sum(best_inliers):
            best_inliers = inliers
            bestH2to1 = H2to1

    return bestH2to1, best_inliers