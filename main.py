import os
import argparse

from utils.extractor import AME


def parse_args():
    parser = argparse.ArgumentParser(prog='AME', description='Activation Maps Extractor')
    group = parser.add_argument_group('Arguments')

    # Required parameters
    group.add_argument('-i', '--image_size', required=True, type=int,
                       help='Input size for test image. If model accept images in format (100, 100, 1) insert 100.')
    group.add_argument('-c', '--channels', required=True, type=int,
                       help='Channels for test image. If model accept images in format (100, 100, 1) insert 1.')
    group.add_argument('-m', '--model_path', required=True, type=str, help='Path to model.')
    group.add_argument('-p', '--image_path', required=True, type=str, help='Path to test image.')
    group.add_argument('-o', '--output_path', required=True, type=str, help='Path for save output.')
    # Optional parameters
    group.add_argument('-e', '--entire_dataset', required=False, type=bool, default=False, help='Perform the extraction'
                                                                                            'on all of the images '
                                                                                            'contained inside the '
                                                                                            'dataset')
    group.add_argument('-n', '--normalize', required=False, type=bool, default=False, help='Normalize image')

    arguments = parser.parse_args()

    return arguments


if __name__ == '__main__':
    arguments = parse_args()

    extractor = AME(
        arguments.image_size,
        arguments.channels,
        arguments.model_path,
        arguments.image_path,
        arguments.output_path,
        arguments.entire_dataset,
        arguments.normalize)

    print('\n[INFO] Start extraction...')

    extractor.extract()

    print('[INFO] Extraction finished.\n')
