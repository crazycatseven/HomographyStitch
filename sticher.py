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

        # find the keypoints with ORB
        keypoints = orb.detect(img_gray, None)

        # compute the descriptors with ORB
        keypoints, descriptors = orb.compute(img_gray, keypoints)

        # convert keypoints to numpy array
        locs = np.array([keypoint.pt for keypoint in keypoints])

    return locs, descriptors


def matchDescriptors(desc_list, ratio=0.8):
    matches = []
    for i in range(len(desc_list) - 1):
        matches.append(skimage.feature.match_descriptors(desc_list[i], desc_list[i + 1], 'hamming', cross_check=True,
                                                         max_ratio=ratio))

    return matches


def computeH(locs_list, matches):
    H_list = []
    for i in range(len(matches)):
        locs1 = locs_list[i][matches[i][:, 0]]
        locs2 = locs_list[i + 1][matches[i][:, 1]]

        homography, _ = planarH.computeH_ransac(locs2, locs1)
        # homography, _ = cv2.findHomography(locs1, locs2, cv2.RANSAC, 2.0)

        H_list.append(homography)

    return H_list


def stitch(img_list, H_list):
    current_img = img_list[0]
    current_translation = [0, 0]
    for i in range(1, 2):
        img1 = current_img
        img2 = img_list[i]
        H = H_list[i-1]

        H_translation0 = np.array([[1, 0, current_translation[0]], [0, 1, current_translation[1]], [0, 0, 1]])

        x1, y1 = img1.shape[:2]
        x2, y2 = img2.shape[:2]

        img1_corners = np.float32([[0, 0], [0, x1], [y1, x1], [y1, 0]]).reshape(-1, 1, 2)
        img2_corners = np.float32([[0, 0], [0, x2], [y2, x2], [y2, 0]]).reshape(-1, 1, 2)

        print()
        print("img1_corners: ", img1_corners)
        print("img2_corners: ", img2_corners)

        img1_corners_transformed = cv2.perspectiveTransform(img1_corners, H_translation0 @ H)
        imgs_corners = np.concatenate((img1_corners_transformed, img2_corners), axis=0)

        [x_min, y_min] = np.int32(imgs_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(imgs_corners.max(axis=0).ravel() + 0.5)

        print('Translation distance: {}'.format([-x_min, -y_min]))

        translation_dist = [-x_min + current_translation[0], -y_min + current_translation[1]]

        print('Translation distance added: {}'.format(translation_dist))

        H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

        img1_warped = cv2.warpPerspective(img1, H_translation.dot(H), (x_max - x_min, y_max - y_min))

        print('img1_warped shape: {}'.format(img1_warped.shape))
        print("translation_dist[0]: ", translation_dist[0])
        print("translation_dist[1]: ", translation_dist[1])
        print("x1 = {}, y1 = {}, x2 = {}, y2 = {}".format(x1, y1, x2, y2))

        output_img = np.zeros_like(img1_warped)

        output_img[translation_dist[1]:x2 + translation_dist[1], translation_dist[0]:y2 + translation_dist[0]] = img2

        # Overlay img1_warped on top of output_img
        mask = img1_warped > 0
        output_img[mask] = img1_warped[mask]
        current_img = output_img
        current_translation = translation_dist

        print('Stitching image {} and {}, output shape {}'.format(i, i + 1, output_img.shape))


    return current_img

# def stitch(img_list, H_list):
#     output_imgs = []
#     for i in range(len(H_list)):
#         img1 = img_list[i]
#         img2 = img_list[i + 1]
#         H = H_list[i]
#
#         rows1, cols1 = img1.shape[:2]
#         rows2, cols2 = img2.shape[:2]
#
#         list_of_points_1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
#         temp_points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)
#
#         # When we have established a homography we need to warp perspective
#         # Change field of view
#         list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
#
#         list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)
#
#         [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
#         [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
#
#         translation_dist = [-x_min, -y_min]
#
#         H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
#
#         img1_warped = cv2.warpPerspective(img1, H_translation.dot(H), (x_max - x_min, y_max - y_min))
#
#         output_img = np.zeros_like(img1_warped)
#
#         output_img[translation_dist[1]:rows1 + translation_dist[1],
#         translation_dist[0]:cols1 + translation_dist[0]] = img2
#
#         # Overlay img1_warped on top of output_img
#         mask = img1_warped > 0
#         output_img[mask] = img1_warped[mask]
#
#         output_imgs.append(output_img)
#
#     return output_imgs
