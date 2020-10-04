import argparse

from loguru import logger
from core.orb import test_orb
from core.sift import test_sift
from core.flann import test_flann
from core.multi_scale import test_multi_scale
from core.functional import read_transparent_png
from core.template_matching import test_template_matching

