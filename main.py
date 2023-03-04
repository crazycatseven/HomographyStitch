import cv2
import matplotlib.pyplot as plt
import sticher
import helper as hlp

if __name__ == '__main__':

    # load the source images in data folder
    imgs = [cv2.imread('data/1.jpg'),
            cv2.imread('data/2.jpg'),
            cv2.imread('data/3.jpg'),
            cv2.imread('data/4.jpg'),
            cv2.imread('data/5.jpg')]

    locs_list = []
    desc_list = []

    # Show the images in the same figure

    plt.figure(figsize=(20, 10))
    for i in range(5):
        method = 'ORB'  # 'BRIEF' or 'ORB', 'BRIEF' is much much slower

        print("Computing keypoints and descriptors for image {}/{}... ".format(i + 1, len(imgs)), end='')
        start = cv2.getTickCount()

        locs, desc = sticher.findKeyPointsAndDescriptors(imgs[i], method=method)
        print("Finished, time: {}s".format((cv2.getTickCount() - start) / cv2.getTickFrequency()))
        locs_list.append(locs)
        desc_list.append(desc)

        # Plot the keypoints on the image

        plt.subplot(2, 5, i + 1)
        plt.axis('off')
        img_to_show = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)
        plt.imshow(img_to_show)

        if method == 'BRIEF':
            plt.scatter(locs[:, 1], locs[:, 0], s=10, marker='.', c='r')
        elif method == 'ORB':
            plt.scatter(locs[:, 0], locs[:, 1], s=10, marker='.', c='r')

    # Match the descriptors
    print("Matching descriptors... ", end='')
    start = cv2.getTickCount()
    matches = sticher.matchDescriptors(desc_list)
    print("Finished, time: {}s".format((cv2.getTickCount() - start) / cv2.getTickFrequency()))

    # plt.show()

    # hlp.plotMatches(imgs[0], imgs[1], matches[0], locs_list[0], locs_list[1])

    # Compute the homography
    print("Computing homography... ", end='')
    start = cv2.getTickCount()
    H_list = sticher.computeH(locs_list, matches)
    print("Finished, time: {}s".format((cv2.getTickCount() - start) / cv2.getTickFrequency()))

    # Stitch the images
    print("Stitching images... ", end='')
    start = cv2.getTickCount()
    stitched_img = sticher.stitch(imgs, H_list)
    print("Finished, time: {}s".format((cv2.getTickCount() - start) / cv2.getTickFrequency()))

    plt.subplot(2, 1, 2)
    plt.axis('off')
    img_to_show = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_to_show)

    plt.show()

    # # Show the stitched image
    # plt.subplot(2, 5, 6)
    #
    # # fig.suptitle('My Big Title', fontsize=20)
    # plt.axis('off')
    # stitched_img0 = cv2.cvtColor(stitched_img[0], cv2.COLOR_BGR2RGB)
    # plt.imshow(stitched_img0)
    #
    # plt.show()

