from copy import deepcopy
from math import pi, ceil

import scipy.ndimage
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
import trimesh
from trimesh import voxel


def load_stl(stl_path):
    return trimesh.load(stl_path)


def stl_to_arr(stl_mesh, pad=True):
    arr = voxel.VoxelMesh(mesh=stl_mesh, pitch=1).matrix_solid
    if pad:
        max_edge_dist = get_max_bbox_edge(stl_mesh)
        arr = center_constant_pad(arr, max_edge_dist)
    return arr


def generate_label(stl_mesh):
    random_angle = np.random.choice(np.arange(-30, 30), 3, replace=True)
    stl_mesh_random_rotated = rotate_stl(
        stl_mesh, random_angle, axes=(1, 1, 1))
    label_3d_arr = stl_to_arr(stl_mesh_random_rotated)
    label_2d_arr = project_3d_arr_to_2d_arr(label_3d_arr)
    return label_3d_arr, label_2d_arr, random_angle


def get_max_bbox_edge(stl_mesh):
    vertices = stl_mesh.bounding_box.vertices
    edge_dists = distance.cdist(vertices, vertices, 'euclidean')
    max_edge_dist = int(edge_dists.max())  # temporary disable
    return 40


def center_constant_pad(arr, target_shape):
    x, y, z = arr.shape
    x_num_pad_left = (target_shape - x) // 2
    x_num_pad_right = ceil((target_shape - x) / 2)
    y_num_pad_left = (target_shape - y) // 2
    y_num_pad_right = ceil((target_shape - y) / 2)
    z_num_pad_left = (target_shape - z) // 2
    z_num_pad_right = ceil((target_shape - z) / 2)
    return np.pad(arr, ((x_num_pad_left, x_num_pad_right), (y_num_pad_left,
        y_num_pad_right), (z_num_pad_left, z_num_pad_right)), 'constant')


def rotate_stl(stl_mesh, angles, axes=(1, 1, 1)):
    stl_mesh_rotated = deepcopy(stl_mesh)
    rads = [i * pi / 180 for i in angles]
    x_rad, y_rad, z_rad = rads
    axis_x, axis_y, axis_z = axes

    if axis_x:
        stl_mesh_rotated.apply_transform(
            trimesh.transformations.rotation_matrix(x_rad,
            (1, 0, 0)))
    if axis_y:
        stl_mesh_rotated.apply_transform(
            trimesh.transformations.rotation_matrix(y_rad,
            (0, 1, 0)))
    if axis_z:
        stl_mesh_rotated.apply_transform(
            trimesh.transformations.rotation_matrix(z_rad,
            (0, 0, 1)))
    return stl_mesh_rotated


def load_3d_img(npy_path):
    arr_3d = np.load(npy_path)
    assert arr_3d.ndim == 3
    return arr_3d


def rotate_3d_arr(arr, rotate=(0, 0, 0)):
    x_angle, y_angle, z_angle = rotate
    arr_x_rotated = scipy.ndimage.interpolation.rotate(
        arr, x_angle, mode="nearest", axes=(1, 2), reshape=False
    )
    arr_y_rotated = scipy.ndimage.interpolation.rotate(
        arr_x_rotated, y_angle, mode="nearest", axes=(0, 2), reshape=False
    )
    arr_z_rotated = scipy.ndimage.interpolation.rotate(
        arr_y_rotated, z_angle, mode="nearest", axes=(0, 1), reshape=False
    )
    return arr_z_rotated


def project_3d_arr_to_2d_arr(arr_3d, axis=-1):
    arr_2d = arr_3d.max(axis)
    return arr_2d


def get_iou(pred_img, target_img):
    union = (pred_img + target_img).sum()
    num_pred_ones = (pred_img).sum()
    num_target_ones = (target_img).sum()
    intersection = num_pred_ones + num_target_ones - union
    iou = intersection / num_target_ones
    return iou


def plot_3d_arr(arr_3d):
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.voxels(arr_3d, edgecolor="k")
    plt.show()


def plot_2d_img(arr_2d):
    plt.figure()
    plt.imshow(arr_2d)
    plt.show()


def plot_stl_mesh(stl_mesh):
    stl_mesh.show()


def unnormalize_action(action):
    return [i * 30 for i in action]


class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        x = np.array(x).reshape(-1, 1, *X.shape)
        y = np.array(y).reshape(-1, 1, *Y.shape)
        u = np.array(u)
        r = np.array(r).reshape(-1, 1)
        d = np.array(d).reshape(-1, 1)
        return x, y, u, r, d
