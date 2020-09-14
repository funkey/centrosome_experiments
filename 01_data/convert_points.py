import csv
import json
import numpy as np
import sys

def load_data(in_file):

    with open(in_file, 'r') as f:
        points = json.load(f)

    return points

def convert_data(
        points,
        point_type='base',
        reverse_axes=False,
        scale=None):

    l = []

    for point in points:
        try:
            p = [float(i) for i in point[point_type]]
            if reverse_axes:
                p = p[::-1]
            if scale:
                p = [i*j for i,j in zip(p,scale)]
            l.append(p)
        except:
            pass

    return l

def write_data(out_file, l):

    with open(out_file, 'w', newline='') as f:
        write = csv.writer(f, delimiter=' ')
        write.writerows(l)

if __name__ == '__main__':

    point_file = load_data(sys.argv[1])

    points = convert_data(
                point_file,
                reverse_axes=True,
                scale=[40,4,4])

    write_data(sys.argv[2], points)
