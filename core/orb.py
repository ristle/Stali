from core.functional import *
from loguru import logger
import numpy as np
import cv2 as cv
import argparse


class Featers:
    def __init__(self):
        logger.info("Init ORB module...", feature="f-strings")

    @staticmethod
    def create_detector():
        detector = cv.ORB_create(nfeatures=2000)

        return detector

    @logger.catch
    def get_features(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        detector = self.create_detector()
        kps, descs = detector.detectAndCompute(gray, None)

        logger.info("Got ORB features...", feature="f-strings")

        return kps, descs, img.shape[:2][::-1]

    def detect_features(self, img, train_features):
        train_kps, train_descs, shape = train_features
        # get features from input image
        kps, descs, _ = self.get_features(img)
        # check if keypoints are extracted
        if not kps:
            logger.error("Not key points!", feature="f-strings")
            return None

        bf = cv.BFMatcher(cv.NORM_HAMMING)
        matches = bf.knnMatch(train_descs, descs, k=2)
        good = []
        # apply ratio test to matches of each keypoint
        # idea is if train KP have a matching KP on image, it will be much closer than next closest non-matching KP,
        # otherwise, all KPs will be almost equally far
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good.append([m])
        # stop if we didn't find enough matching keypoints
        if len(good) < 0.1 * len(train_kps):
            logger.error("Not enough keypoints", feature="f-strings")
            return None

        src_pts = np.float32([train_kps[m[0].queryIdx].pt for m in good
                              ]).reshape(-1, 1, 2)
        dst_pts = np.float32([kps[m[0].trainIdx].pt for m in good
                              ]).reshape(-1, 1, 2)
        m, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        if m is not None:
            scene_points = cv.perspectiveTransform(
                np.float32([(0, 0), (0, shape[0] - 1), (shape[1] - 1, shape[0] - 1), (shape[1] - 1, 0)]).reshape(-1, 1,
                                                                                                                 2),
                m)
            rect = cv.minAreaRect(scene_points)
            if rect[1][1] > 0 and 0.8 < (rect[1][0] / rect[1][1]) < 1.2:
                return rect
        logger.warning("Could not find...")
        return None


@logger.catch
def test_orb(src_filename, dest_filename):
    cv.destroyAllWindows()
    features = Featers()

    img = read_transparent_png(src_filename)
    dest = cv.imread(dest_filename)

    train_features = features.get_features(img)

    region = features.detect_features(dest, train_features)

    if region is not None:
        # draw rotated bounding box
        box = cv.boxPoints(region)
        box = np.int0(box)
        cv.drawContours(dest, [box], 0, (0, 255, 0), 2)

    cv.imshow("ORB", dest)
    cv.waitKey()


'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--src", default='../data/logo.png', help="path for the object image")
    parser.add_argument("--dest", default='../data/Test2.jpg', help="path for image containing the object")
    args = parser.parse_args()

    test_orb(args.src, args.dest)
'''
