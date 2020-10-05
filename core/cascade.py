from core.functional import *
from loguru import logger
import argparse
import cv2
import os

path = os.getcwd()

class Cascade:
    def __init__(self, name=(path + '/data/cascade/cascade.xml')):
        logger.info("Cascade is starting...")
        self.name = name

    @staticmethod
    def create_cascade(name):
        cascade = cv2.CascadeClassifier(str(name))
        logger.info(" created cascade classifier")
        return cascade

    @logger.catch()
    def compute(self, image):
        logger.debug("Name of cascade is {}", self.name)
        cascade = self.create_cascade(self.name)

        logos = cascade.detectMultiScale(image, 50, 50)

        for (x, y, w, h) in logos:
            logger.debug("Drawing detected logos")
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)

        if len(logos) == 0:
            logger.error("No logos was found :c ")

        return image


@logger.catch()
def test_cascade(src_filename, dest_filename):
    cascade = Cascade()

    dest = cv.imread(dest_filename)

    image = cascade.compute(dest)

    cv2.imshow(" Cascade ", image)
    cv2.waitKey()


'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--src", default='../data/logo.png', help="path for the object image")
    parser.add_argument("--dest", default='../data/Test.jpg', help="path for image containing the object")
    args = parser.parse_args()

    test_cascade(args.src, args.dest)
'''