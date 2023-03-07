import cv2
import numpy as np
import skimage.color
import skimage.feature
import helper
import planarH

PATCHWIDTH = 9


def findKeyPointsAndDescriptors(img, method='ORB'):
    """
     Finds keypoints and descriptors in the input image using the specified feature detection and description method.

     Args:
     - img: the input image
     - method: the feature detection and description method to use (default is 'ORB')

     Returns:
     - locs: the locations of the detected keypoints
     - descriptors: the descriptors computed for the detected keypoints
     """
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
        matches.append(skimage.feature.match_descriptors(desc_list[i], desc_list[i + 1], 'hamming', max_ratio=ratio))

    return matches


def computeH(locs1, locs2, matches):
    """
    Computes the homography matrix between two sets of matched points.

    Args:
    - locs1: the keypoints in the first image
    - locs2: the keypoints in the second image
    - matches: the matched keypoints between the two images

    Returns:
    - homography: the computed homography matrix
    """

    # Select the matched keypoints from the input lists
    locs1 = locs1[matches[:, 0]]
    locs2 = locs2[matches[:, 1]]

    # Compute the homography matrix using RANSAC
    homography, _ = planarH.computeH_ransac(locs2, locs1)

    return homography


def stitch_two_image(img1, img2, crop, method='ORB', ratio=0.8):
    """
    Stitches two input images together.
    Note: It will warp img1 to img2's coordinate space.

    Args:
    - img1: the first input image
    - img2: the second input image
    - crop: whether crop the output image to remove black borders (boolean)
    - method: the feature detection and description method to use (default is 'ORB')
    - ratio: the maximum ratio of second-best matches to best matches to consider (default is 0.8)

    Returns:
    - output_img: the stitched output image
    """

    # Compute homography
    locs1, desc1 = findKeyPointsAndDescriptors(img1, method=method)
    locs2, desc2 = findKeyPointsAndDescriptors(img2, method=method)

    matches = skimage.feature.match_descriptors(desc1, desc2, 'hamming', max_ratio=ratio)

    H = computeH(locs1, locs2, matches)

    # Calculate output image size and translation distance
    x1, y1 = img1.shape[:2]
    x2, y2 = img2.shape[:2]

    img1_corners = np.float32([[0, 0], [0, x1], [y1, x1], [y1, 0]]).reshape(-1, 1, 2)
    img2_corners = np.float32([[0, 0], [0, x2], [y2, x2], [y2, 0]]).reshape(-1, 1, 2)

    img1_corners_transformed = cv2.perspectiveTransform(img1_corners, H)
    imgs_corners = np.concatenate((img1_corners_transformed, img2_corners), axis=0)

    [x_min, y_min] = np.int32(imgs_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(imgs_corners.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]

    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    img1_warped = cv2.warpPerspective(img1, H_translation.dot(H), (x_max - x_min, y_max - y_min))

    # Create output image and first place img2 in it
    output_img = np.zeros_like(img1_warped)
    output_img[translation_dist[1]:x2 + translation_dist[1], translation_dist[0]:y2 + translation_dist[0]] = img2

    # Overlay img1_warped on top of output_img
    mask = img1_warped > 0
    output_img[mask] = img1_warped[mask]

    if crop:
        # Calculate the corners of the transformed image
        top_left = np.abs(img1_corners_transformed[0][0] + translation_dist).astype(np.int32)
        bottom_left = np.abs(img1_corners_transformed[1][0] + translation_dist).astype(np.int32)
        bottom_right = np.abs(img1_corners_transformed[2][0] + translation_dist).astype(np.int32)
        top_right = np.abs(img1_corners_transformed[3][0] + translation_dist).astype(np.int32)

        # Calculate the bounding box of the output image
        x1_crop = np.max([top_left[0], bottom_left[0]])
        x2_crop = x1_crop + translation_dist[0] + y2

        y1_crop = np.max([top_left[1], top_right[1]])
        y1_crop = np.max([y1_crop, translation_dist[1]])

        y2_crop = min([bottom_left[1], bottom_right[1], translation_dist[1] + x2])

        # Crop the output image to the calculated bounding box
        output_img = output_img[y1_crop:y2_crop, x1_crop:x2_crop, :]

    return output_img


def stitch_all(img_list, method='ORB', crop=False, ratio=0.8):
    """
    Stitches together all input images in the given list.
    Note: It will warp the first image to the second image's coordinate space,
    then the second image to the third image's coordinate space, etc.

    Args:
    - img_list: a list of input images to stitch together
    - method: the feature detection and description method to use (default is 'ORB')
    - crop: whether crop the output image to remove black borders
    - ratio: the maximum ratio of second-best matches to best matches to consider (default is 0.8)

    Returns:
    - output_img: the stitched output image
    """
    img_list_stitched = []

    # Stitch adjacent pairs of images in the input list
    for i in range(len(img_list) - 1):
        img_list_stitched.append(
            stitch_two_image(img_list[i], img_list[i + 1], method=method, ratio=ratio, crop=crop))

    # Stitch together the results of previous stitching passes until only one image is left
    while len(img_list_stitched) > 1:
        img_list_stitched_new = []
        for i in range(len(img_list_stitched) - 1):
            # Stitch together adjacent pairs of images
            img_list_stitched_new.append(
                stitch_two_image(img_list_stitched[i], img_list_stitched[i + 1], method=method,
                                 ratio=ratio, crop=crop))
        img_list_stitched = img_list_stitched_new

    return img_list_stitched[0]
