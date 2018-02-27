import argparse
import logging

import numpy as np
import cv2


def argparser(parser=None):
    parser = parser or argparse.ArgumentParser(description=__doc__)

    parser.add_argument("file", help="Picture of your document", type=str)
    parser.add_argument('--verbose', '-v', action='count', default=3)

    return parser

def main(args):
    logging.basicConfig(level=max(logging.CRITICAL - (10 * args.verbose), 0))

    img = cv2.imread(args.file)
    logging.info('Image {file} of shape {shape} loaded successfully'.format(file=args.file, shape=img.shape))

    cv2.imshow('Input', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = argparser().parse_args()
    main(args)
