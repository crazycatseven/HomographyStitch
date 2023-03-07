import cv2
import numpy as np
import skimage.color
import skimage.feature
import helper
from functools import reduce
import planarH

PATCHWIDTH = 9


def findKeyPointsAndDescriptors(img, method='BRIEF'):
    img = img.astype(np.uint8)

    # Convert image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    locs = None
    descriptors = None

    if method == 'BRIEF':
        # Detect corners
        corners = helper.corner_detection(img_gray, 5)

        # Compute BRIEF descriptors
        descriptors, locs = helper.computeBrief(img_gray, corners)

    elif method == 'ORB':
        # Initiate ORB detector
        orb = cv2.ORB_create()

        # find the key points with ORB
        keypoints = orb.detect(img_gray, None)

        # compute the descriptors with ORB
        keypoints, descriptors = orb.compute(img_gray, keypoints)

        # convert key points to numpy array
        locs = np.array([keypoint.pt for keypoint in keypoints])

    return locs, descriptors


def matchDescriptors(desc_list, ratio=0.8):
    matches = []
    for i in range(len(desc_list) - 1):
        matches.append(skimage.feature.match_descriptors(desc_list[i], desc_list[i + 1], 'hamming', cross_check=True,
                                                         max_ratio=ratio))

    return matches


def computeH(locs1, locs2, matches):
    locs1 = locs1[matches[:, 0]]
    locs2 = locs2[matches[:, 1]]

    homography, _ = planarH.computeH_ransac(locs2, locs1)
    # homography, _ = cv2.findHomography(locs1, locs2, cv2.RANSAC, 2.0)

    return homography


def stitch_two_image(img1, img2, crop=False, method='ORB', ratio=0.8):
    # Compute homography
    locs1, desc1 = findKeyPointsAndDescriptors(img1, method=method)
    locs2, desc2 = findKeyPointsAndDescriptors(img2, method=method)

    matches = skimage.feature.match_descriptors(desc1, desc2, 'hamming', cross_check=True, max_ratio=ratio)

    H = computeH(locs1, locs2, matches)

    x1, y1 = img1.shape[:2]
    x2, y2 = img2.shape[:2]

    print(x1, y1, x2, y2)

    img1_corners = np.float32([[0, 0], [0, x1], [y1, x1], [y1, 0]]).reshape(-1, 1, 2)
    img2_corners = np.float32([[0, 0], [0, x2], [y2, x2], [y2, 0]]).reshape(-1, 1, 2)

    img1_corners_transformed = cv2.perspectiveTransform(img1_corners, H)
    imgs_corners = np.concatenate((img1_corners_transformed, img2_corners), axis=0)

    [x_min, y_min] = np.int32(imgs_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(imgs_corners.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]

    print(translation_dist)

    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    if not crop:
        img1_warped = cv2.warpPerspective(img1, H_translation.dot(H), (x_max - x_min, y_max - y_min))
        output_img = np.zeros_like(img1_warped)
        output_img[translation_dist[1]:x2 + translation_dist[1], translation_dist[0]:y2 + translation_dist[0]] = img2

        # Overlay img1_warped on top of output_img
        mask = img1_warped > 0
        output_img[mask] = img1_warped[mask]
    else:

        img1_warped = cv2.warpPerspective(img1, H_translation.dot(H), (x1 + x2, y1 + y2))


        top_left = img1_corners_transformed[0][0]
        top_right = img1_corners_transformed[1][0]
        bottom_right = img1_corners_transformed[2][0]
        bottom_left = img1_corners_transformed[3][0]

        x1_crop, y1_crop, x2_crop, y2_crop = None, None, None, None


        if (top_left[0] > bottom_left[0]):


        if (top_left[1] > 0) and (top_left[0] > top_right[0]):
            y1_crop = top_left[1]
        elif (top_right[1] > 0) and (top_right[0] > top_left[0]):
            y1_crop = top_right[1]
        else:
            y1_crop = 0

        if (bottom_left[1] < y2) and (bottom_left[0] < bottom_right[0]):
            y2_crop = bottom_left[1]
        elif (bottom_right[1] < y2) and (bottom_right[0] < bottom_left[0]):
            y2_crop = bottom_right[1]
        else:
            y2_crop = y2




    return output_img


def stitch_all(img_list, method='ORB', ratio=0.8):
    """
    Order 0: Recursively stitch every two images from left to right"
    Order 1: Recursively stitch every two images from right to left"

    """

    img_list_stitched = []

    for i in range(len(img_list) - 1):
        img_list_stitched.append(
            stitch_two_image(img_list[i], img_list[i + 1], method=method, ratio=ratio))

    while len(img_list_stitched) > 1:
        img_list_stitched_new = []
        for i in range(len(img_list_stitched) - 1):
            img_list_stitched_new.append(
                stitch_two_image(img_list_stitched[i], img_list_stitched[i + 1], method=method,
                                 ratio=ratio))
        img_list_stitched = img_list_stitched_new

    return img_list_stitched[0]
