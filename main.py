import cv2
import matplotlib.pyplot as plt
import sticher

# import helper as hlp

if __name__ == '__main__':
    # load the source images in data folder
    imgs = [cv2.imread('data/Room1/1.jpg'),
            cv2.imread('data/Room1/2.jpg'),
            cv2.imread('data/Room1/3.jpg'),
            cv2.imread('data/Room1/4.jpg'),
            cv2.imread('data/Room1/5.jpg')]

    # imgs = [cv2.imread('data/GroupRoom/1.jpg'),
    #         cv2.imread('data/GroupRoom/2.jpg'),
    #         cv2.imread('data/GroupRoom/3.jpg'),
    #         cv2.imread('data/GroupRoom/4.jpg'),
    #         cv2.imread('data/GroupRoom/5.jpg')]

    # Show the images in the same figure
    plt.figure(figsize=(20, 20))

    # room1_output_SIFT_crop = sticher.stitch_all(imgs, method='SIFT', crop=True)
    # room1_output_SIFT = sticher.stitch_all(imgs, method='SIFT', crop=False)
    #
    # room1_output_ORB_crop = sticher.stitch_all(imgs, method='ORB', crop=True)
    # room1_output_ORB = sticher.stitch_all(imgs, method='ORB', crop=False)

    room1_output_BRIEF_crop = sticher.stitch_all(imgs, method='BRIEF', crop=True)
    room1_output_BRIEF = sticher.stitch_all(imgs, method='BRIEF', crop=False)

    # plt.subplot(2, 2, 1)
    # plt.title('SIFT with crop')
    # plt.imshow(cv2.cvtColor(room1_output_SIFT_crop, cv2.COLOR_BGR2RGB))
    #
    # plt.subplot(2, 2, 2)
    # plt.title('SIFT without crop')
    # plt.imshow(cv2.cvtColor(room1_output_SIFT, cv2.COLOR_BGR2RGB))
    #
    # plt.subplot(2, 2, 3)
    # plt.title('ORB with crop')
    # plt.imshow(cv2.cvtColor(room1_output_ORB_crop, cv2.COLOR_BGR2RGB))
    #
    # plt.subplot(2, 2, 4)
    # plt.title('ORB without crop')
    # plt.imshow(cv2.cvtColor(room1_output_ORB, cv2.COLOR_BGR2RGB))

    # plt.show()
