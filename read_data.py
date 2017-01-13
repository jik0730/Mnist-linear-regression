import sys
import struct
import array
import autograd.numpy as np


"""
Description: Read images consisted of bytes. (NOT USED)
Input_1 (str): Filename of byte data.
Input_2 (np.array(?,784)): Pixel matrix initialized by 0s.
"""
def read_images(filename, array):
    with open(filename, "rb") as f:
        magic_number = f.read(4)
        number_of_images = f.read(4)
        number_of_rows = f.read(4)
        number_of_columns = f.read(4)

        byte = f.read(1)
        for x in range(0,int(number_of_images)):
            for y in range(0,number_of_rows*number_of_columns):
                array[x,y] = byte
                byte = f.read(1)


"""
Description: Read labels consisted of bytes. (NOT USED)
Input_1 (str): Filename of byte data.
Input_2 (np.array(?,10)): Label matrix initialized by 0s.
"""
def read_labels(filename, array):
    with open(filename, "rb") as f:
        magic_number = f.read(4)
        number_of_items = f.read(4)

        byte = f.read(1)
        for x in range(0,number_of_items):
            array[x,int(byte)] = 1


def mnist(f1, f2, f3, f4):
    def parse_labels(filename):
        with open(filename, 'rb') as fh:
            magic, num_data = struct.unpack(">II", fh.read(8))
            temp = np.array(array.array("B", fh.read()), dtype=np.uint8)
            toReturn = np.zeros((num_data, 10))
            for i in range(num_data):
                toReturn[i,temp[i]] = 1
            return toReturn

    def parse_images(filename):
        with open(filename, 'rb') as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows*cols)

    train_images = parse_images(f1)
    train_labels = parse_labels(f2)
    test_images  = parse_images(f3)
    test_labels  = parse_labels(f4)

    return train_images, train_labels, test_images, test_labels
