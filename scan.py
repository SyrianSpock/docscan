import argparse
import logging

import numpy as np
import cv2


def argparser(parser=None):
    parser = parser or argparse.ArgumentParser(description=__doc__)

    parser.add_argument("file", help="Picture of your document", type=str)
    parser.add_argument("--debug", action='store_true',
                        help='Display intermediate results')
    parser.add_argument('--verbose', '-v', action='count', default=3)

    return parser

def display(image, title='Debug'):
    cv2.imshow(title, image)
    cv2.waitKey(0)

def load_image(path, debug=False):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    logging.info('Image {file} of shape {shape} loaded successfully'
        .format(file=args.file, shape=img.shape))
    if debug: display(gray)

    return img, gray

def edge_detect(grayscale_image, debug=False):
    edges = cv2.Canny(grayscale_image, 50, 150, apertureSize = 3)
    if debug: display(edges)

    return edges

def find_lines(image, edges_image, debug=False):
    lines = cv2.HoughLines(edges_image, 1, np.pi / 180, 200)
    lines = list(map(lambda line: line[0], lines))
    logging.info('Hough transform found {} lines'.format(len(lines)))

    if debug:
        img_debug = image.copy()
        for rho, theta in lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(img_debug, (x1,y1), (x2,y2), (0,0,255), 2)

        display(img_debug)

    return lines

def main(args):
    logging.basicConfig(level=max(logging.CRITICAL - (10 * args.verbose), 0))

    img, gray = load_image(args.file, debug=args.debug)
    edges = edge_detect(gray, debug=args.debug)
    lines = find_lines(img, edges, debug=args.debug)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = argparser().parse_args()
    main(args)
