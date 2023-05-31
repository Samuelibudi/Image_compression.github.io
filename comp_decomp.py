#!/usr/bin/python3

import cv2
import numpy as np

def compress_image(image_path):
    #load the image
    image = cv2.imread(image_path)
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cb, cr = cv2.split(ycbcr_image)

    padded_height = ((y.shape[0] - 1) // 8 + 1) * 8
    padded_width = ((y.shape[1] - 1) // 8 + 1) * 8

    y_padded = cv2.copyMakeBorder(y, 0, padded_height - y.shape[0], 0, padded_width - y.shape[1], cv2.BORDER_CONSTANT, value=0)
    cb_padded = cv2.copyMakeBorder(cb, 0, padded_height - cb.shape[0], 0, padded_width - cb.shape[1], cv2.BORDER_CONSTANT, value=0)
    cr_padded = cv2.copyMakeBorder(cr, 0, padded_height - cr.shape[0], 0, padded_width - cr.shape[1], cv2.BORDER_CONSTANT, value=0)

    #Define the JPEG quantization matrix
    quantization_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                    [12, 12, 14, 19, 26, 58, 60, 55],
                                    [14, 13, 16, 24, 40, 57, 69, 56],
                                    [14, 17, 22, 29, 51, 87, 80, 62],
                                    [18, 22, 37, 56, 68, 109, 103, 77],
                                    [24, 35, 55, 64, 81, 104, 113, 92],                                                                                                                                                                                [49, 64, 78, 87, 103, 121, 120, 101],
                                    [72, 92, 95, 98, 112, 100, 103, 99]])

    # Function to perform DCT on a block
    def dct_block(block):
        return cv2.dct(np.float32(block))
                    
    # Function to perform quantization on a block
    def quantize_block(block, quantization_matrix):
        return np.round(block / quantization_matrix)

    # Apply DCT and quantization on Y channel
    y_blocks = [
            quantize_block(dct_block(y_padded[i:i + 8, j:j + 8]), quantization_matrix)
            for i in range(0, y_padded.shape[0], 8)
            for j in range(0, y_padded.shape[1], 8)
            ]

    # Save the compressed data
    np.save('compressed_data.npy', {'y_blocks': y_blocks, 'y_padded': y_padded, 'cb': cb_padded, 'cr': cr_padded})        
    print('Image compression completed and compressed data saved as compressed_data.npy.')

def decompress_image():
    # Load the compressed data
    compressed_data = np.load('compressed_data.npy', allow_pickle=True).item()
    # Retrieve the compressed data
    y_blocks = compressed_data['y_blocks']
    y_padded = compressed_data['y_padded']
    cb_padded = compressed_data['cb']
    cr_padded = compressed_data['cr']

    # Define the JPEG quantization matrix
    quantization_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                    [12, 12, 14, 19, 26, 58, 60, 55],
                                    [14, 13, 16, 24, 40, 57, 69, 56],
                                    [14, 17, 22, 29, 51, 87, 80, 62],
                                    [18, 22, 37, 56, 68, 109, 103, 77],
                                    [24, 35, 55, 64, 81, 104, 113, 92],                                                                                                                                                                                [49, 64, 78, 87, 103, 121, 120, 101],
                                    [72, 92, 95, 98, 112, 100, 103, 99]])

    # Function to perform inverse DCT on a block
    def idct_block(block):
        return cv2.idct(block)

    # Function to perform dequantization on a block
    def dequantize_block(block, quantization_matrix):
        return block * quantization_matrix

    # Perform inverse DCT and dequantization on Y channel
    y_reconstructed = np.zeros_like(cb_padded)
    idx = 0
    for i in range(0, y_reconstructed.shape[0], 8):
        for j in range(0, y_reconstructed.shape[1], 8):
            y_block = idct_block(dequantize_block(y_blocks[idx], quantization_matrix))
            y_reconstructed[i:i + 8, j:j + 8] = y_block
            idx += 1

    # Remove padding from the Y channel
    y_original_size = y_padded.shape
    y_reconstructed = y_reconstructed[:y_original_size[0], :y_original_size[1]]

    # Combine YCbCr channels into the reconstructed image
    reconstructed_image = cv2.merge((y_reconstructed, cb_padded, cr_padded))
            
    # Convert the image back to BGR color space
    reconstructed_image = cv2.cvtColor(reconstructed_image, cv2.COLOR_YCrCb2BGR)

    # Save the reconstructed image
    cv2.imwrite('reconstructed_image.jpg', reconstructed_image)
            
    print('Image reconstruction completed and reconstructed image saved as reconstructed_image.jpg.')

# Example usage
compress_image('img1.jpg')
decompress_image()
