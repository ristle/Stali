from core.functional import *
from loguru import logger
import numpy as np
import cv2 as cv
import argparse
import imutils


def multi_scale_template_matching(img_rgb, template):
    logger.info(" Starting multi scale template matching")
    image = img_rgb
    maxVal = 640
    maxLoc = (240, 240)

    # Convert to grayscale
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

    w, h, _ = template.shape[::-1]
    tW, tH = w, h
    found = None

    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(img_gray, width=int(img_gray.shape[1] * scale))
        r = img_gray.shape[1] / float(resized.shape[1])

        found = (maxVal, maxLoc, r)

    # unpack the found varaible and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

    # draw a bounding box around the detected result and display the image
    cv.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
    logger.info("Finished detecting...")
    return image


@logger.catch()
def test_multi_scale(src_filename, dest_filename):
    cv.destroyAllWindows()
    img = read_transparent_png(src_filename)
    dest = cv.imread(dest_filename)

    image = multi_scale_template_matching(dest, img)

    cv.imshow("multi_scale", image)
    cv.waitKey()


'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--src", default='../data/logo.png', help="path for the object image")
    parser.add_argument("--dest", default='../data/Test2.jpg', help="path for image containing the object")
    args = parser.parse_args()

    test_multi_scale(args.src, args.dest)
'''

