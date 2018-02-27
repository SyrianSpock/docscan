import argparse
import logging

import cv2
import numpy as np


def argparser(parser=None):
    parser = parser or argparse.ArgumentParser(description=__doc__)

    parser.add_argument("file", help="Picture of your document", type=str)
    parser.add_argument("--debug", action='store_true',
                        help='Display intermediate results')
    parser.add_argument('--verbose', '-v', action='count', default=3)

    return parser

def configure_logging(level):
    logging.basicConfig(level=max(logging.CRITICAL - (10 * level), 0))

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

def draw_line(image, rho, theta, color=(0,0,255), thickness=2):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    cv2.line(image, (x1,y1), (x2,y2), color, thickness)

def find_lines(image, edges_image, debug=False):
    lines = cv2.HoughLines(edges_image, 1, np.pi / 180, 200)
    lines = list(map(lambda line: line[0], lines))
    logging.info('Hough transform found {} lines'.format(len(lines)))

    if debug:
        img_debug = image.copy()
        for rho, theta in lines:
            draw_line(img_debug, rho, theta)
        display(img_debug)

    return lines

def segment_by_angle(image, lines, debug=False):
    # map line slop to points on a unit circle
    angles = np.array([line[1] for line in lines])
    points = np.array([[np.cos(2 * angle), np.sin(2 * angle)]
                       for angle in angles], dtype=np.float32)

    compactness, labels, centers = cv2.kmeans(
        data=points, K=2, bestLabels=None,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)

    labels = labels.reshape(-1)  # transpose to row vec
    labeled_lines = list(zip(lines, labels))
    lines_by_label = lambda label: [line[0] for line in labeled_lines if line[1] == label]
    segmented = {label: lines_by_label(label) for label in labels}
    logging.info('Segmented {} lines into {} labels'.format(len(lines), len(set(labels))))

    if debug:
        img_debug = image.copy()
        colors = [(0, 255 * label / max(labels), 255) for label in labels]
        for label in segmented:
            for line in segmented[label]:
                rho, theta = line
                draw_line(img_debug, rho, theta, color=colors[label])
        display(img_debug)

    return list(segmented.values())

def main(args):
    configure_logging(args.verbose)
    img, gray = load_image(args.file, debug=args.debug)
    edges = edge_detect(gray, debug=args.debug)
    lines = find_lines(img, edges, debug=args.debug)
    segmented = segment_by_angle(img, lines, debug=args.debug)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = argparser().parse_args()
    main(args)
