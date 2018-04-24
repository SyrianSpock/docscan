import argparse
import logging

import cv2
import numpy as np
from PIL import Image
import pytesseract

PAPER_SIZES = {
    'A4' : (2100, 2970) # 10 pixels per mm
}

def argparser(parser=None):
    parser = parser or argparse.ArgumentParser(description=__doc__)

    parser.add_argument("file", help="Picture of your document", type=str)
    parser.add_argument("--debug", action='store_true',
                        help='Display intermediate results')
    parser.add_argument('--verbose', '-v', action='count', default=3)
    parser.add_argument('--paper-size', '-p', type=str, default='A4')

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

def intersection(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    return np.linalg.solve(A, b)

def draw_cross(image, pos, size=10, color=(0, 0, 255), thickness=2):
    top = (pos[0], int(pos[1] - size / 2))
    down = (pos[0], int(pos[1] + size / 2))
    left = (int(pos[0] - size / 2), pos[1])
    right = (int(pos[0] + size / 2), pos[1])

    cv2.line(image, top, down, color, thickness)
    cv2.line(image, left, right, color, thickness)

def segmented_intersections(image, lines, debug=False):
    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            intersections += [intersection(l1, l2) for l1 in group for l2 in next_group]
    logging.info('Found {} intersections'.format(len(intersections)))

    if debug:
        img_debug = image.copy()
        for point in intersections:
            draw_cross(img_debug, point)
        display(img_debug)

    return intersections

def _closest_point(point, points):
    distance = [np.linalg.norm(p.flatten() - point.flatten()) for p in points]

    closest_point_index = np.argmin(distance)
    closest_point = points[closest_point_index]

    logging.debug('Found nearest to {} at {} : {}'
                    .format(point, closest_point_index, closest_point.flatten()))

    return closest_point_index, closest_point

def image_corners(shape):
    return np.array([
        [0, 0], # topleft
        [0, shape[0]], # topright
        [shape[1], shape[0]], # bottomright
        [shape[1], 0], # bottomleft
    ], dtype=np.float32)

def find_document_corners(image, points, debug=False):
    document_corners = np.array([
        _closest_point(corner, points)[1].flatten()
        for corner in image_corners(image.shape)
    ])

    if debug:
        img_debug = image.copy()
        for point in points:
            draw_cross(img_debug, point)
        for point in document_corners:
            draw_cross(img_debug, point, color=(0, 255, 0))
        display(img_debug)

    return document_corners

def undistort_document(image, corners, output_shape, debug=False):
    transform = cv2.getPerspectiveTransform(corners, image_corners(output_shape))
    doc = cv2.warpPerspective(image, transform, (output_shape[1], output_shape[0]))

    if debug:
        img_debug = doc.copy()
        display(img_debug)

    return doc

def main(args):
    configure_logging(args.verbose)
    img, gray = load_image(args.file, debug=args.debug)
    edges = edge_detect(gray, debug=args.debug)
    lines = find_lines(img, edges, debug=args.debug)
    segmented = segment_by_angle(img, lines, debug=args.debug)
    intersections = segmented_intersections(img, segmented, debug=args.debug)

    corners = find_document_corners(img, intersections, debug=args.debug)
    document = undistort_document(img, corners, img.shape, debug=args.debug)

    gray = undistort_document(gray, corners, img.shape, debug=args.debug)

    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(10,10))
    gray = clahe.apply(gray)

    img = Image.fromarray(gray).resize(PAPER_SIZES[args.paper_size])

    txt = pytesseract.image_to_string(img, lang='fra', config='--oem=2 --psm=2').encode('utf-8')
    print(txt)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = argparser().parse_args()
    main(args)
