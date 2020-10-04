from core import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--src", default='./data/logo.png', help="path for the object image")
    parser.add_argument("--dest", default='./data/Test2.jpg', help="path for image containing the object")
    parser.add_argument("--method", default="all",
                        help="Method for finding logo. Types :\n -all\n -orb\n -flann\n -multi_scale\n -matching \n -test_sift")
    args = parser.parse_args()

    if args.method == "all":
        test_orb(args.src, args.dest)
        test_sift(args.src, args.dest)
        test_flann(args.src, args.dest)
        test_multi_scale(args.src, args.dest)
        test_template_matching(args.src, args.dest)
    elif args.method == "orb":
        test_orb(args.src, args.dest)
    elif args.method == "flann":
        test_flann(args.src, args.dest)
    elif args.method == "multi_scale":
        test_multi_scale(args.src, args.dest)
    elif args.method == "matching":
        test_template_matching(args.src, args.dest)
    elif args.method == "test_sift":
        test_sift(args.src, args.dest)
    else:
        logger.error(" --method error")
    logger.info("All Done")