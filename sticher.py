import cv2
import numpy as np
import skimage.color
import skimage.feature
import helper
import planarH
# import matplotlib.pyplot as plt

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

    elif method == 'SIFT':
        sift=cv2.SIFT_create()

        # find the key points with SIFT
        keypoints, descriptors = sift.detectAndCompute(img_gray, None)

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


def blendingMask(height, width, barrier, smoothing_window, left_biased=True):
    assert barrier < width
    mask = np.zeros((height, width))

    offset = int(smoothing_window/2)
    try:
        if left_biased:
            mask[:,barrier-offset:barrier+offset+1]=np.tile(np.linspace(1,0,2*offset+1).T, (height, 1))
            mask[:,:barrier-offset] = 1
        else:
                mask[:,barrier-offset:barrier+offset+1]=np.tile(np.linspace(0,1,2*offset+1).T, (height, 1))
                mask[:,barrier+offset:] = 1
    except:
        if left_biased:
                mask[:,barrier-offset:barrier+offset+1]=np.tile(np.linspace(1,0,2*offset).T, (height, 1))
                mask[:,:barrier-offset] = 1
        else:
            mask[:,barrier-offset:barrier+offset+1]=np.tile(np.linspace(0,1,2*offset).T, (height, 1))
            mask[:,barrier+offset:] = 1

    return cv2.merge([mask, mask, mask])

def blending(img1,img2,img2_width,side):
    """
    Blending two input images together.

    Args:
    - img1: the first input image (Src)
    - img2: the second input image (Dst)
    - img2_width: the width of img2
    - side: which side is img1 on


    Returns:
    - output_img: the stitched output image
    """   
    h,w,_=img2.shape
    smoothing_window=int(img2_width/8)
    border = img2_width-int(smoothing_window/2)
    mask1 = blendingMask(h, w, border, smoothing_window = smoothing_window, left_biased = True)
    mask2 = blendingMask(h, w, border, smoothing_window = smoothing_window, left_biased = False)

    if side=='left':
        img2=cv2.flip(img2,1)
        img1=cv2.flip(img1,1)
        img2=(img2*mask1)
        img1=(img1*mask2)
        output=img1+img2
        output=cv2.flip(output,1)
    else:
        img2=(img2*mask1)
        img1=(img1*mask2)
        output=img1+img2
    return output

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

    matches = skimage.feature.match_descriptors(desc1, desc2, max_ratio=ratio)

    H = computeH(locs1, locs2, matches)

    # Calculate output image size and translation distance
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    img1_corners = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    img2_corners = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    try:
        img1_corners_transformed = cv2.perspectiveTransform(img1_corners, H)
        imgs_corners = np.concatenate((img1_corners_transformed, img2_corners), axis=0)

        [x_min, y_min] = np.int32(imgs_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(imgs_corners.max(axis=0).ravel() + 0.5)

        translation_dist = [-x_min, -y_min]
        
        # Determine whether img1 is on the left side or right side of the output image
        # if the top left corner (Transformed) have x < 0, then it should be on the left side
        if(imgs_corners[0][0][0]<0):
            side='left'
            # width_output=w2+translation_dist[0]
        else:
            # width_output = int(img1_corners_transformed[3][0][0])
            side='right'
        width_output=x_max-x_min
        height_output=y_max-y_min

        H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
        img1_warped = cv2.warpPerspective(img1, H_translation.dot(H), (width_output,height_output))

        # # Create output image
        output_img = np.zeros_like(img1_warped)
        
        # Generating size of img2_resized which has the same size as img1_warped
        img2_resized=np.zeros((height_output,width_output,3),dtype="uint8")
        if side=='left':
            img2_resized[translation_dist[1]:h2+translation_dist[1],translation_dist[0]:w2+translation_dist[0]] = img2
        else:
            img2_resized[translation_dist[1]:h2+translation_dist[1],:w2] = img2

        # Blending
        output_img=np.asarray(blending(img1_warped,img2_resized,w2,side),dtype="uint8")
    
    except:
        raise Exception("The image set doesn't meet the requirement.")

    if crop:
        left_border=0
        right_border=width_output
        if side=="left":
            left_border=int(np.max([img1_corners_transformed[0][0][0],img1_corners_transformed[1][0][0]])+translation_dist[0])
        
        else:
            right_border=int(np.min([img1_corners_transformed[2][0][0],img1_corners_transformed[3][0][0]])+translation_dist[0])
        
        top_border=int(np.max([img1_corners_transformed[0][0][1],
                               img1_corners_transformed[3][0][1],
                               img2_corners[0][0][0]])+translation_dist[1])
        bottom_border=int(np.min([img1_corners_transformed[1][0][1]+translation_dist[1],
                                  img1_corners_transformed[2][0][1]+translation_dist[1],
                                  img2_corners[1][0][1]])+translation_dist[1])
        output_img=output_img[top_border:bottom_border,left_border:right_border,:]
    
    # Only For Test
    # plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    # plt.show()

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

    n=int(len(img_list)/2+0.5)
    left=img_list[:n]
    right=img_list[n-1:]
    right.reverse()
    while len(left)>1:
        dst_img=left.pop()
        src_img=left.pop()
        left_pano=stitch_two_image(src_img,dst_img, method=method, ratio=ratio, crop=crop)
        left.append(left_pano)

    while len(right)>1:
        dst_img=right.pop()
        src_img=right.pop()
        right_pano=stitch_two_image(src_img,dst_img, method=method, ratio=ratio, crop=crop)
        right.append(right_pano)
    
    #if width_right_pano > width_left_pano, Select right_pano as destination. Otherwise is left_pano
    if(right_pano.shape[1]>=left_pano.shape[1]):
        fullpano=stitch_two_image(left_pano,right_pano,method=method, ratio=ratio, crop=crop)
    else:
        fullpano=stitch_two_image(right_pano,left_pano,method=method, ratio=ratio, crop=crop)
    
    return fullpano
