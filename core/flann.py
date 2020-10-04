from core.functional import *
from loguru import logger
import numpy as np
import cv2 as cv
import argparse


class Flann:
    def __init__(self):
        logger.info("Started Flann method")

    @staticmethod
    def create_detector():
        orb = cv.ORB_create(nfeatures=500)
        logger.info("Inited detector")

        return orb

    @logger.catch()
    def flann_mathes(self, image1, image2):
        index_params = dict(algorithm=6,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=2)
        search_params = {}
        flann = cv.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(image1, image2, k=2)
        logger.info("Computed  flann...")
        return matches

    @logger.catch()
    def correct_image(self, img1, img2, kp1, kp2, good_matches):
        src_points = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_points = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        m, mask = cv.findHomography(src_points, dst_points, cv.RANSAC, 5.0)
        corrected_img = cv.warpPerspective(img1, m, (img2.shape[1], img2.shape[0]))

        return corrected_img

    @logger.catch()
    def get_corrected_img(self, img1, img2):
        MIN_MATCHES = 50

        orb = self.create_detector()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        matches = self.flann_mathes(des1, des2)

        # As per Lowe's ratio test to filter good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) > MIN_MATCHES:
            corrected_img = self.correct_image(img1, img2, kp1, kp2, good_matches)

            logger.info("Got corrected image")
            return corrected_img

        logger.warning("No matching with Flann method")
        return img2

@logger.catch()
def test_flann(src_filename, dest_filename):
    cv.destroyAllWindows()
    flann = Flann()

    img = read_transparent_png(src_filename)
    dest = cv.imread(dest_filename)

    corrected_image = flann.get_corrected_img(img, dest)

    cv.imshow("FLANN", corrected_image)
    cv.waitKey()


'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--src", default='../data/logo.png', help="path for the object image")
    parser.add_argument("--dest", default='../data/Test.jpg', help="path for image containing the object")
    args = parser.parse_args()

    test_flann(args.src, args.dest)
'''
