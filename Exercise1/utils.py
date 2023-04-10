import numpy
import numpy as np
from PIL import Image
# from numba import jit
# from tqdm import tqdm
from abc import abstractmethod, abstractstaticmethod
import time
from copy import deepcopy


class SeamImage:
    def __init__(self, img_path, vis_seams=True):
        """ SeamImage initialization.

        Parameters:
            img_path (str): image local path
            method (str) (a or b): a for Hard Vertical and b for the known Seam Carving algorithm
            vis_seams (bool): if true, another version of the original image shall be store, and removed seams should be marked on it
        """
        #################
        # Do not change #
        #################
        self.path = img_path

        self.gs_weights = np.array([[0.299, 0.587, 0.114]]).T

        self.rgb = self.load_image(img_path)
        self.resized_rgb = self.rgb.copy()
        self.gs_squeezed = None
        self.dx, self.dy = None, None

        self.vis_seams = vis_seams
        if vis_seams:
            self.seams_rgb = self.rgb.copy()

        self.h, self.w = self.rgb.shape[:2]

        try:
            self.gs = self.rgb_to_grayscale(self.rgb)
            self.resized_gs = self.gs.copy()
            self.cumm_mask = np.ones_like(self.gs, dtype=bool)
        except NotImplementedError as e:
            print(e)

        try:
            self.E = self.calc_gradient_magnitude()
        except NotImplementedError as e:
            print(e)
        #################

        # additional attributes you might find useful
        self.seam_history = None
        self.seam_balance = 0
        self.seam_history_vertical = []
        self.seam_history_horizontal = []

        # This might serve you to keep tracking original pixel indices
        self.idx_map_h, self.idx_map_v = np.meshgrid(range(self.w), range(self.h))

    def rgb_to_grayscale(self, np_img):
        """ Converts a np RGB image into grayscale (using self.gs_weights).
        Parameters
            np_img : ndarray (float32) of shape (h, w, 3)
        Returns:
            grayscale image (float32) of shape (h, w, 1)

        Guidelines & hints:
            Use NumpyPy vectorized matrix multiplication for high performance.
            To prevent outlier values in the boundaries, we recommend to pad them with 0.5
        """

        grey_img = np.dot(np_img, self.gs_weights)
        grey_img = np.pad(grey_img, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0.5)
        self.gs_squeezed = grey_img.squeeze()
        return grey_img

    def calc_gradient_magnitude(self):
        """ Calculate gradient magnitude of a grayscale image

        Returns:
            A gradient magnitude image (float32) of shape (h, w)

        Guidelines & hints:
            In order to calculate a gradient of a pixel, only its neighborhood is required.
        """

        gs = self.gs_squeezed
        self.dx = np.diff(gs, axis=0)[1:, 1:-1]
        self.dy = np.diff(gs, axis=1)[1:-1, 1:]
        pixel_energy = np.sqrt(np.square(self.dx) + np.square(self.dy))

        """
        start = time.time()
        rows = self.h
        columns = self.w
        pixel_energy = np.zeros((rows, columns))
        for row in range(rows):
            for column in range(columns):

                if column < columns - 1:
                    e_ver = abs(self.gs[row, column] - self.gs[row, column + 1])
                else:
                    e_ver = abs(self.gs[row, column] - self.gs[row, column - 1])

                if row < rows - 1:
                    e_hor = abs(self.gs[row, column] - self.gs[row + 1, column])
                else:
                    e_hor = abs(self.gs[row, column] - self.gs[row - 1, column])
                pixel_energy[row, column] = np.sqrt(np.square(e_ver) + np.square(e_hor))
        end = time.time()
        print(f"öld way: {end-start}")
        """

        return pixel_energy

    def calc_M(self):
        pass

    def seams_removal(self, num_remove):
        pass

    def seams_removal_horizontal(self, num_remove):
        pass

    def seams_removal_vertical(self, num_remove):
        pass

    def rotate_mats(self, clockwise):
        pass

    def init_mats(self):
        pass

    def update_ref_mat(self):
        pass

    def backtrack_seam(self):
        pass

    def remove_seam(self):
        pass

    def reinit(self):
        """ re-initiates instance
        """
        self.__init__(self.path)

    @staticmethod
    def load_image(img_path):
        return np.asarray(Image.open(img_path)).astype('float32') / 255.0


class ColumnSeamImage(SeamImage):
    """ Column SeamImage.
    This class stores and implements all required data and algorithmics from implementing the "column" version of the seam carving algorithm.
    """

    def __init__(self, *args, **kwargs):
        """ ColumnSeamImage initialization.
        """
        super().__init__(*args, **kwargs)
        try:
            self.M = self.calc_M()
            self.M_copy = np.copy(self.M)
        except NotImplementedError as e:
            print(e)

    # TODO: NIR
    def calc_M(self):
        """ Calculates the matrix M discussed in lecture, but with the additional constraint:
            - A seam must be a column. That is, the set of seams S is simply columns of M.
            - implement forward-looking cost

        Returns:
            A "column" energy matrix M (float32) of shape (h, w)

        Guidelines & hints:
            As taught, the energy is calculated from top to bottom.
            The formula of calculation M is as taught, but with certain terms omitted.
            You might find the function 'np.roll' useful.
        """
        gs = self.gs_squeezed
        cost_vertical = np.abs(np.roll(gs, 1, axis=1) - np.roll(gs, -1, axis=1))[1:-1, 1:-1]
        m_costs = self.E + cost_vertical
        m_costs = np.cumsum(m_costs, axis=0)
        return m_costs

    # TODO: ORI
    def seams_removal(self, num_remove: int):
        """ Iterates num_remove times and removes num_remove vertical seams

        Parameters:
            num_remove (int): number of vertical seam to be removed

        Guidelines & hints:
        As taught, the energy is calculated from top to bottom.
        You might find the function np.roll useful.

        This step can be divided into a couple of steps:
            i) init/update matrices (E, M, backtracking matrix, saem mask) where:
                - E is the gradient magnitude matrix
                - M is the cost matric
                - backtracking matrix is an idx matrix used to track the minimum seam from bottom up
                - mask is a boolean matrix for removed seams
            ii) seam backtracking: calculates the actual indices of the seam
            iii) index update: when a seam is removed, index mapping should be updated in order to keep track indices for next iterations
            iv) seam removal: create the carved image with the reduced (and update seam visualization if desired)
            Note: the flow described below is a recommendation. You may implement seams_removal as you with, but it needs to support:
            - removing seams couple of times (call the function more than once)
            - visualize the original image with removed seams marked (for comparison)
        """

        for i in range(num_remove):
            min_index = np.argmin(self.M[-1, :])
            removed_before_rel = sum(self.seam_history <= min_index)
            cand = min_index + removed_before_rel
            while cand in self.seam_history:
                cand += 1
            min_index_abs = cand

            self.seam_balance += 1
            self.seam_history.append(min_index_abs)

            self.update_E(min_index)
            self.update_M(min_index)
            self.backtrack_seam(min_index_abs)
            self.remove_seam(min_index)

        # raise NotImplementedError("TODO: Implement SeamImage.seams_removal")

    def update_E(self, seam_idx):
        # Remove the seam at the given index
        (rows, columns) = self.E.shape
        self.E = np.delete(self.E, seam_idx, axis=1)
        self.dx = np.delete(self.dx, seam_idx, axis=1)
        self.dy = np.delete(self.dy, seam_idx, axis=1)
        gs = self.gs_squeezed

        seam_idx_in_padded_img = seam_idx + 1
        if 0 < seam_idx <= columns - 1:
            self.dy[:, seam_idx - 1] = (gs[:, seam_idx_in_padded_img - 1] - gs[:, seam_idx_in_padded_img + 1])[1:-1]
            self.E[:, seam_idx - 1] = np.sqrt(np.square(self.dy[:, seam_idx - 1]) + np.square(self.dx[:, seam_idx - 1]))

    def update_M(self, seam_idx):
        # Remove the seam at the given index
        (rows, columns) = self.M.shape
        self.M = np.delete(self.M, seam_idx, axis=1)
        gs = self.gs_squeezed
        seam_idx_in_padded_img = seam_idx + 1

        if seam_idx > 0:
            edges_cost_left = np.abs(gs[:, seam_idx_in_padded_img - 2] - gs[:, seam_idx_in_padded_img + 1])[1:-1]
            self.M[:, seam_idx - 1] = np.cumsum(self.E[:, seam_idx - 1] + edges_cost_left)
        if seam_idx < columns - 1:
            edges_cost_right = np.abs(gs[:, seam_idx_in_padded_img + 2] - gs[:, seam_idx_in_padded_img - 1])[1:-1]
            self.M[:, seam_idx] = np.cumsum(self.E[:, seam_idx] + edges_cost_right)

    def seams_removal_horizontal(self, num_remove):
        """ Removes num_remove horizontal seams

        Parameters:
            num_remove (int): number of horizontal seam to be removed

        Guidelines & hints:
            You may find np.rot90 function useful

        """
        self.seam_history = self.seam_history_horizontal

        self.resized_gs = np.rot90(self.resized_gs, 1, (0, 1))
        self.resized_rgb = np.rot90(self.resized_rgb, 1, (0, 1))
        self.gs_squeezed = np.rot90(self.gs_squeezed, 1, (0, 1))
        self.dx, self.dy = np.rot90(self.dy, 1, (0, 1)), np.rot90(self.dx, 1, (0, 1))
        self.E = np.rot90(self.E, 1, (0, 1))
        self.cumm_mask = np.rot90(self.cumm_mask, 1, (0, 1))
        self.seams_rgb = np.rot90(self.seams_rgb, 1, (0, 1))
        self.M = self.calc_M()  # we must calculate M anew since we don't have horizontal-edges cost information

        self.seams_removal(num_remove)
        self.resized_gs = np.rot90(self.resized_gs, 3, (0, 1))
        self.resized_rgb = np.rot90(self.resized_rgb, 3, (0, 1))
        self.gs_squeezed = np.rot90(self.gs_squeezed, 3, (0, 1))
        self.dx, self.dy = np.rot90(self.dy, 3, (0, 1)), np.rot90(self.dx, 3, (0, 1))
        self.E = np.rot90(self.E, 3, (0, 1))
        self.M = self.calc_M()  # revive "vertical" M
        self.cumm_mask = np.rot90(self.cumm_mask, 3, (0, 1))
        self.seams_rgb = np.rot90(self.seams_rgb, 3, (0, 1))

        self.seam_history = None

        # raise NotImplementedError("TODO: Implement SeamImage.seams_removal_horizontal")

    def seams_removal_vertical(self, num_remove):
        """ A wrapper for removing num_remove horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): number of vertical seam to be removed
        """
        self.seam_history = self.seam_history_vertical
        self.seams_removal(num_remove)
        self.seam_history = None

    def backtrack_seam(self, seam_idx):
        """ Backtracks a seam for Column Seam Carving method
        """
        seam_idx_in_padded_img = seam_idx + 1
        self.cumm_mask[:, seam_idx_in_padded_img] = False
        self.seams_rgb[:, seam_idx_in_padded_img, :] = [1, 0, 0]

    def remove_seam(self, seam_idx):
        """ Removes a seam for self.rgb (you may create a resized version, like self.resized_rgb)

        Guidelines & hints:
        In order to apply the removal, you might want to extend the seam mask to support 3 channels (rgb) using: 3d_mak = np.stack([1d_mask] * 3, axis=2), and then use it to create a resized version.
        """
        # TODO: change this method
        seam_idx_in_padded_img = seam_idx + 1
        self.gs_squeezed = np.delete(self.gs_squeezed, seam_idx_in_padded_img, axis=1)
        self.resized_gs = np.delete(self.resized_gs, seam_idx_in_padded_img, axis=1)
        self.resized_rgb = np.delete(self.resized_rgb, seam_idx_in_padded_img, axis=1)


class VerticalSeamImage(SeamImage):
    def __init__(self, *args, **kwargs):
        """ VerticalSeamImage initialization.
        """
        super().__init__(*args, **kwargs)
        try:
            self.M = self.calc_M()
        except NotImplementedError as e:
            print(e)

    def calc_M(self):
        """ Calculates the matrix M discussed in lecture (with forward-looking cost)

        Returns:
            An energy matrix M (float32) of shape (h, w)

        Guidelines & hints:
            As taught, the energy is calculated from top to bottom.
            You might find the function 'np.roll' useful.
        """

        # cost_matrix = deepcopy(self.E)
        # minimum_cost = None
        # gs = self.gs_squeezed
        # for row in range(1, self.E.shape[0]):
        #     left_step_value = np.roll(gs[row - 1], 1)
        #     middle_step = gs[row - 1] #np.abs(np.roll(gs[row - 1], 1) - np.roll(gs[row - 1], -1))
        #     right_step_value = np.roll(gs[row - 1], -1)
        #
        #     minimum_cost = np.minimum(left_step_value, middle_step)
        #     minimum_cost = np.minimum(minimum_cost, right_step_value)
        #     cost_matrix[row] = self.E[row] + minimum_cost[1:-1]
        #
        # cost_matrix = np.cumsum(cost_matrix, axis=0)
        # return cost_matrix

        r, c, p = self.gs.shape

        m_cost = self.E.copy()
        # backtrack = np.zeros_like(M, dtype=np.int)

        for i in range(1, r - 2):
            for j in range(0, c - 2):
                # Handle the left edge of the image, to ensure we don't index -1
                if j == 0:
                    idx = np.argmin(m_cost[i - 1, j:j + 2])
                    # backtrack[i, j] = idx + j
                    min_energy = m_cost[i - 1, idx + j]
                else:
                    idx = np.argmin(m_cost[i - 1, j - 1:j + 2])
                    # backtrack[i, j] = idx + j - 1
                    min_energy = m_cost[i - 1, idx + j - 1]

                m_cost[i, j] += min_energy

        return m_cost #, backtrack


    def seams_removal(self, num_remove: int):
        """ Iterates num_remove times and removes num_remove vertical seams

        Parameters:
            num_remove (int): number of vertical seam to be removed

        Guidelines & hints:
        As taught, the energy is calculated from top to bottom.
        You might find the function np.roll useful.

        This step can be divided into a couple of steps:
            i) init/update matrices (E, M, backtracking matrix, saem mask) where:
                - E is the gradient magnitude matrix
                - M is the cost matrix
                - backtracking matrix is an idx matrix used to track the minimum seam from bottom up
                - mask is a boolean matrix for removed seams
            ii) fill in the backtrack matrix corresponding to M
            iii) seam backtracking: calculates the actual indices of the seam
            iv) index update: when a seam is removed, index mapping should be updated in order to keep track indices for next iterations
            v) seam removal: create the carved image with the reduced (and update seam visualization if desired)
            Note: the flow described below is a recommendation. You may implement seams_removal as you with, but it needs to supprt:
            - removing seams couple of times (call the function more than once)
            - visualize the original image with removed seams marked (for comparison)
        """

        for i in range(num_remove):
            min_index = np.argmin(self.M[-1, :])
            removed_before_rel = sum(self.seam_history <= min_index)
            cand = min_index + removed_before_rel
            while cand in self.seam_history:
                cand += 1
            min_index_abs = cand

            self.seam_balance += 1
            self.seam_history.append(min_index_abs)

            self.update_E(min_index)
            self.update_M(min_index)
            self.backtrack_seam(min_index_abs)
            self.remove_seam(min_index)

        raise NotImplementedError("TODO: Implement SeamImage.seams_removal")

    def seams_removal_horizontal(self, num_remove):
        """ Removes num_remove horizontal seams

        Parameters:
            num_remove (int): number of horizontal seam to be removed

        Guidelines & hints:
            You may find np.rot90 function useful

        """
        raise NotImplementedError("TODO: Implement SeamImage.seams_removal_horizontal")

    def seams_removal_vertical(self, num_remove):
        """ A wrapper for removing num_remove horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): umber of vertical seam to be removed
        """

        raise NotImplementedError("TODO: Implement SeamImage.seams_removal_vertical")

    def backtrack_seam(self):
        """ Backtracks a seam for Seam Carving as taught in lecture
        """
        raise NotImplementedError("TODO: Implement SeamImage.backtrack_seam_b")

    def remove_seam(self):
        """ Removes a seam from self.rgb (you may create a resized version, like self.resized_rgb)

        Guidelines & hints:
        In order to apply the removal, you might want to extend the seam mask to support 3 channels (rgb) using: 3d_mak = np.stack([1d_mask] * 3, axis=2), and then use it to create a resized version.
        """
        raise NotImplementedError("TODO: Implement SeamImage.remove_seam")

    def seams_addition(self, num_add: int):
        """ BONUS: adds num_add seamn to the image

            Parameters:
                num_add (int): number of horizontal seam to be removed

            Guidelines & hints:
            - This method should be similar to removal
            - You may use the wrapper functions below (to support both vertical and horizontal addition of seams)
            - Visualization: paint the added seams in green (0,255,0)

        """
        raise NotImplementedError("TODO: Implement SeamImage.seams_addition")

    def seams_addition_horizontal(self, num_add):
        """ A wrapper for removing num_add horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): number of horizontal seam to be added

        Guidelines & hints:
            You may find np.rot90 function useful

        """
        raise NotImplementedError("TODO (Bonus): Implement SeamImage.seams_addition_horizontal")

    def seams_addition_vertical(self, num_add):
        """ A wrapper for removing num_add vertical seams (just a recommendation)

        Parameters:
            num_add (int): number of vertical seam to be added
        """

        raise NotImplementedError("TODO (Bonus): Implement SeamImage.seams_addition_vertical")

    @staticmethod
    # @jit(nopython=True)
    def calc_bt_mat(M, backtrack_mat):
        """ Fills the BT back-tracking index matrix. This function is static in order to support Numba. To use it, uncomment the decorator above.

        Recommnded parameters (member of the class, to be filled):
            M: np.ndarray (float32) of shape (h,w)
            backtrack_mat: np.ndarray (int32) of shape (h,w): to be filled here

        Guidelines & hints:
            np.ndarray is a rederence type. changing it here may affected outsde.
        """
        raise NotImplementedError("TODO: Implement SeamImage.calc_bt_mat")


def scale_to_shape(orig_shape: np.ndarray, scale_factors: list):
    """ Converts scale into shape

    Parameters:
        orig_shape (np.ndarray): original shape [y,x]
        scale_factors (list): scale factors for y,x respectively

    Returns
        the new shape
    """
    raise NotImplementedError("TODO: Implement SeamImage.scale_to_shape")


def resize_seam_carving(seam_img: SeamImage, shapes: np.ndarray):
    """ Resizes an image using Seam Carving algorithm

    Parameters:
        seam_img (SeamImage) The SeamImage instance to resize
        shapes (np.ndarray): desired shape (y,x)

    Returns
        the resized rgb image
    """
    raise NotImplementedError("TODO: Implement SeamImage.resize_seam_carving")


def bilinear(image, new_shape):
    """
    Resizes an image to new shape using bilinear interpolation method
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :returns: the image resized to new_shape
    """
    in_height, in_width, _ = image.shape
    out_height, out_width = new_shape
    new_image = np.zeros(new_shape)

    ###Your code here###
    def get_scaled_param(org, size_in, size_out):
        scaled_org = (org * size_in) / size_out
        scaled_org = min(scaled_org, size_in - 1)
        return scaled_org

    scaled_x_grid = [get_scaled_param(x, in_width, out_width) for x in range(out_width)]
    scaled_y_grid = [get_scaled_param(y, in_height, out_height) for y in range(out_height)]
    x1s = np.array(scaled_x_grid, dtype=int)
    y1s = np.array(scaled_y_grid, dtype=int)
    x2s = np.array(scaled_x_grid, dtype=int) + 1
    x2s[x2s > in_width - 1] = in_width - 1
    y2s = np.array(scaled_y_grid, dtype=int) + 1
    y2s[y2s > in_height - 1] = in_height - 1
    dx = np.reshape(scaled_x_grid - x1s, (out_width, 1))
    dy = np.reshape(scaled_y_grid - y1s, (out_height, 1))
    c1 = np.reshape(image[y1s][:, x1s] * dx + (1 - dx) * image[y1s][:, x2s], (out_width, out_height, 3))
    c2 = np.reshape(image[y2s][:, x1s] * dx + (1 - dx) * image[y2s][:, x2s], (out_width, out_height, 3))
    new_image = np.reshape(c1 * dy + (1 - dy) * c2, (out_height, out_width, 3)).astype(int)
    return new_image


import numba
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from PIL import Image
from numba import jit

plt.rcParams["figure.figsize"] = (10, 5)
# %%
# img_path = "pinguins.jpg"
img_path = "sunset.JPG"


# helper functions
def read_image(img_path):
    return np.asarray(Image.open(img_path)).astype('float32')


def show_image(np_img, grayscale=False):
    fig, ax = plt.subplots()
    if not grayscale:
        ax.imshow(np_img)
    else:
        ax.imshow(np_img, cmap=plt.get_cmap('gray'))
    ax.axis("off")
    plt.show()


def init_plt_grid(nrow=1, ncols=1, figsize=(20, 10), **kwargs):
    fig, ax = plt.subplots(nrow, ncols, figsize=figsize, facecolor='gray', **kwargs)
    font_size = dict(size=20)
    return ax, font_size


### 1. Imaplement SeamImage (20 points)
# %%
# TODO: Create a SeamImage instance

s_img = SeamImage(img_path)

# display sample image we will be working on
# show_image(s_img.rgb)

# disply grayscale version
# show_image(s_img.gs, grayscale=True)

# display its energy (gradient magnitude)
# show_image(s_img.E, grayscale=True)
print("here")
# TODO: Create a ColumnSeamImage instance
cs_img = ColumnSeamImage(img_path)
# %%
# dispay matrices
ax, font_size = init_plt_grid(ncols=3, figsize=(20, 10))

ax[0].set_title('Original Image', **font_size)
ax[1].set_title('Gradient Magnitude Image (E)', **font_size)
ax[2].set_title('Energy Matric (M)', **font_size)

ax[0].imshow(cs_img.rgb)
ax[1].imshow(cs_img.E, cmap='gray')
ax[2].imshow(cs_img.M, cmap="gray")

# dp_mat = vs_img.calc_dp_mat(vs_img.M.copy(), np.zeros_like(vs_img.M, dtype=int))
# ax[1,1].imshow(dp_mat)

for sp in ax.reshape(-1):
    sp.set_xticks([])
    sp.set_yticks([])

plt.tight_layout()
plt.show()

cs_img.seams_removal_vertical(70)
cs_img.seams_removal_horizontal(110)

cs_img_2 = ColumnSeamImage(img_path)
cs_img_2.seams_removal_vertical(30)
cs_img_2.seams_removal_horizontal(50)
cs_img_2.seams_removal_vertical(40)
cs_img_2.seams_removal_horizontal(60)

ax, font_size = init_plt_grid(ncols=3, figsize=(20, 10))

ax[0].set_title('Original Image', **font_size)
ax[1].set_title('Seam Visualization', **font_size)
ax[2].set_title('Resized Image', **font_size)

ax[0].imshow(cs_img.rgb)
ax[1].imshow(cs_img.seams_rgb)
ax[2].imshow(cs_img.resized_rgb)

# save resized version
Image.fromarray((cs_img.resized_rgb * 255).astype('uint8')).save('./resized_rgb.jpg')

for sp in ax.reshape(-1):
    sp.set_xticks([])
    sp.set_yticks([])

plt.tight_layout()
plt.show()

# vs_img = VerticalSeamImage(img_path)
# #%%
# # dispay matrices
#
# ax, font_size = init_plt_grid(ncols=3)
#
# ax[0].set_title('Original Image', **font_size)
# ax[1].set_title('Gradient Magnitude Image (E)', **font_size)
# ax[2].set_title('Cost Matrix (M)', **font_size)
#
# ax[0].imshow(vs_img.rgb)
# ax[1].imshow(vs_img.E, cmap='gray')
# ax[2].imshow(vs_img.M)
#
# for sp in ax.reshape(-1):
#     sp.set_xticks([])
#     sp.set_yticks([])
#
# plt.tight_layout()
# # remove seams (modify as you with)
# vs_img.seams_removal_vertical(70)
# vs_img.seams_removal_horizontal(110)
# #%%
# # display resulting images
#
# ax, font_size = init_plt_grid(ncols=3, figsize=(20,10))
#
# ax[0].set_title('Original Image', **font_size)
# ax[1].set_title('Seam Visualization', **font_size)
# ax[2].set_title('Resized Image', **font_size)
#
# ax[0].imshow(vs_img.rgb)
# ax[1].imshow(vs_img.seams_rgb)
# ax[2].imshow(vs_img.resized_rgb)
#
# # save resized version
# Image.fromarray((vs_img.resized_rgb*255).astype('uint8')).save('./resized_rgb.jpg')
#
# for sp in ax.reshape(-1):
#     sp.set_xticks([])
#     sp.set_yticks([])
#
# plt.tight_layout()
# #%%
# # Resluts comparison
#
# ax, font_size = init_plt_grid(2, 2, figsize=(12,12))
#
# ax[0,0].set_title('SV-SC - Seams', **font_size)
# ax[1,0].set_title('V-SC - Seams', **font_size)
# ax[0,1].set_title('SV-SC - Resized', **font_size)
# ax[1,1].set_title('V-SC - Resized', **font_size)
#
# ax[0,0].imshow(cs_img.seams_rgb)
# ax[1,0].imshow(vs_img.seams_rgb)
# ax[0,1].imshow(cs_img.resized_rgb)
# ax[1,1].imshow(vs_img.resized_rgb)
#
# for sp in ax.reshape(-1):
#     sp.set_xticks([])
#     sp.set_yticks([])
#
# plt.tight_layout()
