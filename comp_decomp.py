#!/usr/bin/python3

import cv2
import numpy as np
import os
import sys


def compress_image(image_path):
    # load the image
    image = cv2.imread(image_path)
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cb, cr = cv2.split(ycbcr_image)

    padded_height = ((y.shape[0] - 1) // 8 + 1) * 8
    padded_width = ((y.shape[1] - 1) // 8 + 1) * 8

    y_padded = cv2.copyMakeBorder(y, 0, padded_height - y.shape[0], 0,
                                  padded_width - y.shape[1],
                                  cv2.BORDER_CONSTANT, value=0)
    cb_padded = cv2.copyMakeBorder(cb, 0, padded_height - cb.shape[0], 0,
                                   padded_width - cb.shape[1],
                                   cv2.BORDER_CONSTANT, value=0)
    cr_padded = cv2.copyMakeBorder(cr, 0, padded_height - cr.shape[0], 0,
                                   padded_width - cr.shape[1],
                                   cv2.BORDER_CONSTANT, value=0)

    quantization_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                    [12, 12, 14, 19, 26, 58, 60, 55],
                                    [14, 13, 16, 24, 40, 57, 69, 56],
                                    [14, 17, 22, 29, 51, 87, 80, 62],
                                    [18, 22, 37, 56, 68, 109, 103, 77],
                                    [24, 35, 55, 64, 81, 104, 113, 92],
                                    [49, 64, 78, 87, 103, 121, 120, 101],
                                    [72, 92, 95, 98, 112, 100, 103, 99]])

    def dct_block(block):
        # Function to perform DCT on a block
        return cv2.dct(np.float32(block))

    def quantize_block(block, quantization_matrix):
        # Function to perform quantization on a block
        return np.round(block / quantization_matrix)

    y_blocks = [
            quantize_block(dct_block(y_padded[i:i + 8, j:j + 8]),
                           quantization_matrix)
            for i in range(0, y_padded.shape[0], 8)
            for j in range(0, y_padded.shape[1], 8)
            ]

    compressed_data = {'y_blocks': y_blocks, 'y_padded': y_padded,
                       'cb': cb_padded, 'cr': cr_padded}

    return compressed_data


def decompress_image(compressed_data):
    # Load the compressed data and compress it.
    y_blocks = compressed_data['y_blocks']
    y_padded = compressed_data['y_padded']
    cb_padded = compressed_data['cb']
    cr_padded = compressed_data['cr']

    quantization_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                    [12, 12, 14, 19, 26, 58, 60, 55],
                                    [14, 13, 16, 24, 40, 57, 69, 56],
                                    [14, 17, 22, 29, 51, 87, 80, 62],
                                    [18, 22, 37, 56, 68, 109, 103, 77],
                                    [24, 35, 55, 64, 81, 104, 113, 92],
                                    [49, 64, 78, 87, 103, 121, 120, 101],
                                    [72, 92, 95, 98, 112, 100, 103, 99]])

    def idct_block(block):
        # Function to perfomr inverse DCT on a block
        return cv2.idct(block)

    def dequantize_block(block, quantization_matrix):
        # Function to perform dequantisation on a block
        return block * quantization_matrix

    y_reconstructed = np.zeros_like(cb_padded)
    idx = 0
    for i in range(0, y_reconstructed.shape[0], 8):
        for j in range(0, y_reconstructed.shape[1], 8):
            y_block = idct_block(dequantize_block
                                 (y_blocks[idx], quantization_matrix))
            y_reconstructed[i:i + 8, j:j + 8] = y_block
            idx += 1

    y_original_size = y_padded.shape
    y_reconstructed = y_reconstructed[:y_original_size[0], :y_original_size[1]]
    reconstructed_image = cv2.merge((y_reconstructed, cb_padded, cr_padded))
    reconstructed_image = cv2.cvtColor(reconstructed_image,
                                       cv2.COLOR_YCrCb2BGR)


    print('Image reconstruction completed')
    return reconstructed_image

if len(sys.argv) == 2:
    filepath = sys.argv[1]
else:
    print("USAGE: image_compression + <filepath>")


base_filename = os.path.basename(filepath)
rec_img = decompress_image(compress_image(filepath))
cv2.imwrite('./compressed_images/'+base_filename, rec_img)
