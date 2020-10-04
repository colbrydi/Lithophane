"""
This is the Lithophane Module written by Dirk Colbry.

Core of this module uses matlab-stl to write stl 
files written by Rick van Hattem. 

"""

import matplotlib.image as img
import matplotlib.pyplot as plt
import os
import sys
#from PIL import Image
from skimage.transform import resize

import numpy as np

from mpl_toolkits import mplot3d
from matplotlib import pyplot

from stl import mesh


def rgb2gray(rgb):
    """Convert rgb image to grayscale image in range 0-1

    >>> gray = factorial(rgbimg)

    """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def scaleim(im, width_mm=40):
    """Scale image to 0.1 pixel width
    For example the following:

    >>> im_scaled = scaleim(im, width_mm = 100)

    Will make an image with 1000 pixels wide.
    The height will be scale proportionally
    """

    ydim = im.shape[0]
    xdim = im.shape[1]

    scale = (width_mm*10/xdim)
    newshape = (int(ydim*scale), int(xdim*scale), 3)
    im = resize(im, newshape)
    return im


def jpg2stl(im='', width='', h=3.0, d=0.5, show=True):
    """Function to convert filename to stl with width = width

    :width: - Required parameter.  Width

    """
    depth = h
    offset = d

    if type(im) == str:
        filename = im
        print(f"Reading {filename}")
        im = img.imread(filename)
    else:
        filenmae = 'image.xxx'

    if width == '':
        width = im.shape[1]

    # TODO: Width is actually height
    im = scaleim(im, width_mm=width)

    im = im/np.max(im)

    # Convert to grayscale
    if len(im.shape) == 3:
        gray = rgb2gray(im)
    else:
        gray = im

    #g = np.fliplr(g)
    if(show):
        plt.imshow(gray, cmap=plt.get_cmap('gray'))

    # print(np.max(g))
    # print(g.shape)

    # Invert threshold for z matrix
    ngray = 1 - np.double(gray)

    # scale z matrix to desired max depth and add base height
    z_middle = ngray * depth + offset

    # add border of zeros to help with back.
    z = np.zeros([z_middle.shape[0]+2, z_middle.shape[1]+2])
    z[1:-1, 1:-1] = z_middle

    x1 = np.linspace(1, z.shape[1]/10, z.shape[1])
    y1 = np.linspace(1, z.shape[0]/10, z.shape[0])

    x, y = np.meshgrid(x1, y1)

    x = np.fliplr(x)

    return x, y, z


def makeCylinder(x, y, z):
    '''Convert flat point cloud to Cylinder'''
    newx = x.copy()
    newz = z.copy()
    radius = (np.max(x)-np.min(x))/(2*np.pi)
    print(f"Cylinder Radius {radius}mm")
    for r in range(0, x.shape[0]):
        for c in range(0, x.shape[1]):
            t = (c/(x.shape[1]-10))*2*np.pi
            rad = radius + z[r, c]
            newx[r, c] = rad*np.cos(t)
            newz[r, c] = rad*np.sin(t)
    return newx, y.copy(), newz

# Construct polygons from grid data


def makemesh(x, y, z):
    '''Convert point cloud grid to mesh'''
    count = 0
    points = []
    triangles = []
    for i in range(z.shape[0]-1):
        for j in range(z.shape[1]-1):

            # Triangle 1
            points.append([x[i][j], y[i][j], z[i][j]])
            points.append([x[i][j+1], y[i][j+1], z[i][j+1]])
            points.append([x[i+1][j], y[i+1][j], z[i+1][j]])

            triangles.append([count, count+1, count+2])

            # Triangle 2
            points.append([x[i][j+1], y[i][j+1], z[i][j+1]])
            points.append([x[i+1][j+1], y[i+1][j+1], z[i+1][j+1]])
            points.append([x[i+1][j], y[i+1][j], z[i+1][j]])

            triangles.append([count+3, count+4, count+5])

            count += 6

    # BACK
    for j in range(x.shape[1]-1):

        bot = x.shape[0]-1

        # Back Triangle 1
        points.append([x[bot][j], y[bot][j], z[bot][j]])
        points.append([x[0][j+1], y[0][j+1], z[0][j+1]])
        points.append([x[0][j],   y[0][j],   z[0][j]])

        triangles.append([count, count+1, count+2])

        # Triangle 2
        points.append([x[bot][j], y[bot][j], z[bot][j]])
        points.append([x[bot][j+1], y[bot][j+1], z[bot][j+1]])
        points.append([x[0][j+1], y[0][j+1], z[0][j+1]])

        triangles.append([count+3, count+4, count+5])

        count += 6

    # Create the mesh
    model = mesh.Mesh(np.zeros(len(triangles), dtype=mesh.Mesh.dtype))
    for i, f in enumerate(triangles):
        for j in range(3):
            model.vectors[i][j] = points[f[j]]

    return model


def makeSphere(x, y, z, radius=None, bottom_radian=0.85):
    '''Convert flat point cloud to Sphere'''
    r_max = int(x.shape[0] * (1 - np.arccos(bottom_hole_radius / radius) / np.pi))
    copy_target = slice(r_max)
    front_x = x[copy_target].copy()
    front_y = y[copy_target].copy()
    front_z = z[copy_target].copy()
    back_x = x[copy_target].copy()
    back_y = y[copy_target].copy()
    back_z = z[copy_target].copy()
    for r in range(0, r_max):
        p = float(r)/x.shape[0]*np.pi
        for c in range(0, front_x.shape[1]):
            t = (c/(x.shape[1]-10))*2*np.pi
            external_radius = radius + z[r, c] if r < r_max - 1 else radius + thickness
            internal_radius = radius
            front_x[r, c] = external_radius*np.cos(t)*np.sin(p)
            front_y[r, c] = external_radius*np.cos(p)
            front_z[r, c] = external_radius*np.sin(t)*np.sin(p)
            back_x[r, c] = internal_radius*np.cos(t)*np.sin(p)
            back_y[r, c] = internal_radius*np.cos(p) if r < r_max - 1 else front_y[r, c]
            back_z[r, c] = internal_radius*np.sin(t)*np.sin(p)
    return (front_x, front_y, front_z), (back_x, back_y, back_z)


def makeMeshSphere(front, back):
    '''Convert point cloud grid to mesh'''
    x, y, z = front
    bx, by, bz = back
    count = 0
    points = []
    triangles = []
    for i in range(x.shape[0]-1):
        for j in range(x.shape[1]-1):

            # Triangle 1
            points.append([x[i][j], y[i][j], z[i][j]])
            points.append([x[i][j+1], y[i][j+1], z[i][j+1]])
            points.append([x[i+1][j], y[i+1][j], z[i+1][j]])

            triangles.append([count, count+1, count+2])

            # Triangle 2
            points.append([x[i][j+1], y[i][j+1], z[i][j+1]])
            points.append([x[i+1][j+1], y[i+1][j+1], z[i+1][j+1]])
            points.append([x[i+1][j], y[i+1][j], z[i+1][j]])

            triangles.append([count+3, count+4, count+5])

            count += 6

    # Back
    for i in range(bx.shape[0]-1):
        for j in range(bx.shape[1]-1):

            # Triangle 1
            points.append([bx[i+1][j], by[i+1][j], bz[i+1][j]])
            points.append([bx[i][j+1], by[i][j+1], bz[i][j+1]])
            points.append([bx[i][j], by[i][j], bz[i][j]])

            triangles.append([count, count+1, count+2])

            # Triangle 2
            points.append([bx[i+1][j], by[i+1][j], bz[i+1][j]])
            points.append([bx[i+1][j+1], by[i+1][j+1], bz[i+1][j+1]])
            points.append([bx[i][j+1], by[i][j+1], bz[i][j+1]])

            triangles.append([count+3, count+4, count+5])

            count += 6

    # Bottom
    bottom_i = bx.shape[0] - 1
    for j in range(bx.shape[1]-1):
        # Triangle 1
        points.append([x[bottom_i][j], y[bottom_i][j], z[bottom_i][j]])
        points.append([bx[bottom_i][j], by[bottom_i][j], bz[bottom_i][j]])
        points.append([bx[bottom_i][j+1], by[bottom_i][j+1], bz[bottom_i][j+1]])

        triangles.append([count, count+1, count+2])

        # Triangle 2
        points.append([x[bottom_i][j], y[bottom_i][j], z[bottom_i][j]])
        points.append([x[bottom_i][j+1], y[bottom_i][j+1], z[bottom_i][j+1]])
        points.append([bx[bottom_i][j+1], by[bottom_i][j+1], bz[bottom_i][j+1]])

        triangles.append([count+3, count+4, count+5])

        count += 6

    # Create the mesh
    model = mesh.Mesh(np.zeros(len(triangles), dtype=mesh.Mesh.dtype))
    for i, f in enumerate(triangles):
        for j in range(3):
            model.vectors[i][j] = points[f[j]]

    return model


def showstl(x, y, z):
    '''
    ======================
    3D surface (color map)
    ======================

    Demonstrates plotting a 3D surface colored with the coolwarm color map.
    The surface is made opaque by using antialiased=False.

    Also demonstrates using the LinearLocator and custom formatting for the
    z axis tick labels.
    '''

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import numpy as np

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # plt.axis('equal')


if __name__ == "__main__":
    import sys
    jpg2stl(sys.argv[2])
