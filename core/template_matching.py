from matplotlib import pyplot as plt
from core.functional import *
from loguru import logger
import numpy as np
import cv2 as cv
import argparse


def template_matching(img_rgb, template, meth='cv.TM_SQDIFF_NORMED'):
    # Convert to grayscale
    img = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    img.astype(np.uint8)

    w, h, _ = template.shape[::-1]

    template = cv.cvtColor(template, cv.COLOR_RGB2GRAY)
    template.astype(np.uint8)

    method = eval(meth)

    logger.info("Performing matching...")
    # Apply template Matching
    res = cv.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv.rectangle(img_rgb, top_left, bottom_right, 255, 2)

    return img_rgb


def test_template_matching(src_filename, dest_filename, meth='cv.TM_SQDIFF_NORMED'):
    cv.destroyAllWindows()
    img = read_transparent_png(src_filename)
    dest = cv.imread(dest_filename)

    corrected_image = template_matching(dest, img, meth=meth)

    cv.imshow('template_matching', corrected_image)
    cv.waitKey()


'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--src", default='../data/logo.png', help="path for the object image")
    parser.add_argument("--dest", default='../data/Test.jpg', help="path for image containing the object")
    args = parser.parse_args()

    test_template_matching(args.src, args.dest)
'''
