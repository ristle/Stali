import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
from core.functional import *
from loguru import logger
import cv2 as cv
import argparse


class SIFT:
    def __init__(self, debug=True):
        figure, self.ax = plt.subplots(1, 2, figsize=(16, 8))
        logger.info("Starting SIFT module")
        self.debug = debug

    @staticmethod
    def create_detector():
        sift = cv.SIFT_create()
        logger.info("Created SIFT Detector")

        return sift

    @logger.catch()
    def sift_detect(self, sift, logo, positive):
        keypoints_1, descriptors_1 = sift.detectAndCompute(logo, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(positive, None)

        # for debugging
        if self.debug:
            logger.debug("Drawing keypoints 1 {}", len(keypoints_1))
            img_1 = cv.drawKeypoints(logo, keypoints_1, logo)
            self.ax[0].imshow(img_1)

            logger.debug("Drawing keypoints 2 {}", len(keypoints_2))
            img_2 = cv.drawKeypoints(positive, keypoints_2, positive)
            self.ax[1].imshow(img_2)

        return keypoints_1, descriptors_1, keypoints_2, descriptors_2

    @logger.catch()
    def flann(self, descriptors_1, descriptors_2):
        flann = cv.FlannBasedMatcher(dict(algorithm=0, tree=5), {})

        matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)  ## 2 кластера нужно по идее
        # matches = sorted(matches, key = lambda x : x.distance) ## это не для knn матчеров было

        if self.debug:
            logger.debug("Len of matches {}", len(matches))

        return matches

    @staticmethod
    def determine_matches(matches):
        goodMatch = list()
        for m, n in matches:
            if m.distance < 0.75 * n.distance:  ## хз насчет этой метрики
                goodMatch.append(m)
        if len(goodMatch) < 3:
            logger.warning("Small count of good matches")

        return goodMatch

    @logger.catch()
    def compute(self, logo, positive):
        img_1, img_2 = logo, positive

        aug = iaa.Resize(256)
        # positive = aug.augment_image(positive)
        logo = aug.augment_image(logo)

        sift = self.create_detector()
        keypoints_1, descriptors_1, keypoints_2, descriptors_2 = self.sift_detect(sift, logo, positive)

        matches = self.flann(descriptors_1, descriptors_2)
        goodMatch = self.determine_matches(matches)
        if self.debug:
            logger.debug("Len on goodMatch is {}", len(goodMatch))

        logger.info("Ended sift...")
        logger.info("Drawing best matches")

        img3 = cv.drawMatchesKnn(img_1, keypoints_1, img_2, keypoints_2, matches, img_2, flags=2)
        plt.imshow(img3)
        plt.show()
        cv.waitKey()


@logger.catch()
def test_sift(src_filename, dest_filename):
    sift = SIFT()

    img_no_correction = read_transparent_png(src_filename)
    img = cv.GaussianBlur(img_no_correction, (3, 3), 2)
    dest = cv.imread(dest_filename)

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dest = cv.cvtColor(dest, cv.COLOR_BGR2GRAY)

    dest = cv.GaussianBlur(dest, (7, 7), 4)
    sift.compute(img, dest)


'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--src", default='../data/logo.png', help="path for the object image")
    parser.add_argument("--dest", default='../data/Test.jpg', help="path for image containing the object")
    args = parser.parse_args()

    test_sift(args.src, args.dest)
'''
