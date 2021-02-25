#!/usr/bin/python3
import argparse
import os
import glob
import random
import time

import cv2
import numpy as np

import darknet


def parser():
    parser = argparse.ArgumentParser(description='YOLO Object Detection')
    parser.add_argument('config_file', type=str, help='Path to config file')
    parser.add_argument('data_file', type=str, help='Path to data file')
    parser.add_argument('weights', type=str, help='Path to weights file')
    parser.add_argument(
        'input', type=str, help='image source. It can be a single image, a txt with paths to them, or a folder. Valid formats are jpg, jpeg or png.')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='number of images to be processed at the same time')
    parser.add_argument('--thresh', type=float, default=0.15,
                        help='remove detections with lower confidence')
    parser.add_argument('--take', type=int, default=None,
                        help='Number of samples to process')

    parser.add_argument('--save_to', type=str,
                        default=None, help='Export path')
    parser.add_argument('--save_labels', action='store_true',
                        help='Save detections for each image. Requires --save_to arg to be set.')
    parser.add_argument('--save_image', action='store_true',
                        help='Save image with detections. Requires --save_to arg to be set.')
    parser.add_argument('--print_detections', action='store_true',
                        help='Print list of detections')
    return parser.parse_args()


def check_args(args):
    assert 0 < args.thresh < 1, 'Threshold should be within [0, 1)'
    if not os.path.exists(args.config_file):
        raise ValueError(
            f'Invalid config path {os.path.abspath(args.config_file)}.')
    if not os.path.exists(args.weights):
        raise ValueError(
            f'Invalid weight path {os.path.abspath(args.weights)}.')
    if not os.path.exists(args.data_file):
        raise ValueError(
            f'Invalid data file path {os.path.abspath(args.data_file)}.')
    if args.input and not os.path.exists(args.input):
        raise ValueError(f'Invalid image path {os.path.abspath(args.input)}.')
    if args.save_image or args.save_labels:
        assert args.save_to is not None


def draw_boxes(detections, image, colors):
    import cv2
    for label, confidence, bbox in detections:
        left, top, right, bottom = darknet.bbox2points(bbox)
        cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)
        # cv2.putText(image, f'{label} ({round(float(confidence))}%)',
        #             (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[label], 2)
    return image


def scale_detection(detection, scale):
    label, confidence, bbox = detection
    x, y, w, h = bbox
    x *= scale[0]
    y *= scale[1]
    w *= scale[0]
    h *= scale[1]
    bbox = x, y, w, h
    return label, confidence, bbox


def scale_detections(detections, scale):
    detections_scaled = []
    for detection in detections:
        detections_scaled.append(scale_detection(detection, scale))
    return detections_scaled


def check_shapes(images, batch_size):
    shapes = [image.shape for image in images]
    if len(set(shapes)) > 1:
        raise ValueError("Images don't have same shape")
    if len(shapes) > batch_size:
        raise ValueError('Batch size higher than number of images')
    return shapes[0]


def load_images(images_path, take):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """
    filenames = []
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        filenames.append(images_path)
    elif input_path_extension == 'txt':
        with open(images_path, 'r') as f:
            filenames.extend(f.read().splitlines())
    else:
        filenames.extend(glob.glob(os.path.join(images_path, '*.jpg')))
        filenames.extend(glob.glob(os.path.join(images_path, '*.png')))
        filenames.extend(glob.glob(os.path.join(images_path, '*.jpeg')))
    filenames = sorted(filenames)
    if take:
        filenames = filenames[:take]

    return filenames


def prepare_batch(network, images, channels=3):
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    batch_size = len(images)
    darknet_image = darknet.make_image(width, height, channels * batch_size)
    images_resized = []
    for image in images:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        images_resized.append(image_resized)
    image_resized = np.concatenate(images_resized, axis=-1)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    return darknet_image


def batch_detection(network, images, class_names, class_colors,
                    thresh=0.25, hier_thresh=.5, nms=.45, batch_size=4):
    height, width, _ = check_shapes(images, batch_size)
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    scale = [images[0].shape[1] / width, images[0].shape[0] / height]
    darknet_images = prepare_batch(network, images)
    batch_detections = darknet.network_predict_batch(network, darknet_images, batch_size, width,
                                                     height, thresh, hier_thresh, None, 0, 0)
    darknet.free_image(darknet_images)
    batch_predictions = []
    for idx in range(batch_size):
        num = batch_detections[idx].num
        detections = batch_detections[idx].dets
        if nms:
            darknet.do_nms_sort(detections, num, len(class_names), nms)
        detections = darknet.remove_negatives(detections, class_names, num)
        detections = scale_detections(detections, scale)
        images[idx] = draw_boxes(
            detections, images[idx], class_colors)
        batch_predictions.append(detections)
    darknet.free_batch_detections(batch_detections, batch_size)
    return images, batch_predictions


def convert_abs2rel(image, bbox):
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x/width, y/height, w/width, h/height


def save_labels(fn, image, detections, class_names):
    with open(fn, 'w') as f:
        for label, confidence, bbox in detections:
            x, y, w, h = convert_abs2rel(image, bbox)
            label = class_names.index(label)
            f.write(
                f'{label} {x:.4f} {y:.4f} {w:.4f} {h:.4f} {float(confidence):.4f}\n')


def create_batches(inputs, batch_size):
    outputs = []
    for idx in range(0, len(inputs), batch_size):
        outputs.append(inputs[idx:idx+batch_size])
    last_batch = outputs[-1]
    last_input = last_batch[-1]

    # Replicate last element to fill up last batch
    for _ in range(batch_size - len(last_batch)):
        last_batch.append(last_input)
    outputs[-1] = last_batch
    return outputs


def detect(args):
    if args.save_to and (args.save_image or args.save_labels):
        print(f'Saving output to {os.path.abspath(args.save_to)}')
        os.makedirs(args.save_to, exist_ok=True)

    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=args.batch_size
    )

    filenames = load_images(args.input, args.take)
    batch_image_names = create_batches(filenames, args.batch_size)

    for count, image_names in enumerate(batch_image_names):
        batch_images = [cv2.imread(image_name) for image_name in image_names]

        t_start = time.time()
        batch_images, batch_detections = batch_detection(network, batch_images, class_names,
                                                         class_colors, thresh=args.thresh, batch_size=args.batch_size)
        t_end = time.time()
        print(
            f'{count} / {len(batch_image_names)} @ {int(args.batch_size/(t_end - t_start))} fps')
        for image_name, image, detections in zip(image_names, batch_images, batch_detections):
            path_out = os.path.join(args.save_to,
                                    os.path.basename(image_name)).split('.')[:-1][0]
            if args.save_image:
                cv2.imwrite(filename=path_out + '.png', img=image)
            if args.print_detections:
                print(detections)
            if args.save_labels:
                save_labels(path_out + '.txt', image, detections, class_names)


if __name__ == '__main__':
    args = parser()
    check_args(args)
    random.seed(1)  # custom colors
    detect(args)
