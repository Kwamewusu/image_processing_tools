#!/usr/bin/env python

##  Mathematical Morphology Masking Routine
# Author: Nana Owusu
# 
# Purpose: To provide Python routines for
# generating masks of loaded grayscale images 
# by way of mathematical morphology.
# These methods can be found in the first part of the
# RATS algorithm by Oguz et al.
# 
# RATS algorithm: https://www.iibi.uiowa.edu/rats-overview
# methods: https://www.sciencedirect.com/science/article/abs/pii/S0165027013003440


# Module for showing the image
from matplotlib.pyplot import subplots, show

# Module for image manipulation
from numpy import where, rot90, zeros, bool, indices, stack, ones

# Module for image morphology operators
from scipy.ndimage import binary_erosion, binary_dilation, grey_erosion, \
    generate_binary_structure, binary_fill_holes, label, sum, convolve1d
from scipy.signal.windows import general_gaussian

def multi_slice_display(input_vol):
    """Display all the slices of a single
    timepoint"""
    
    _, _, o = input_vol.shape
    
    # Instantiate a Figure object with one row
    # and multiple columns of axes.
    fig, axes = subplots(1, o, figsize=[28, 35])
    fig.subplots_adjust(wspace=0.125)
    
    # Fill the figure with each slice
    for i,ax in enumerate(axes.flat):
        ax.imshow(rot90(input_vol[:,:,i]), cmap='gray')
        ax.set_axis_off()
        
    show

class struct_el(object):
    
    def __init__(self):
        self.shape = \
        {'square': self.square,
         'cube': self.cube,
         'circle': self.circle,
         'sphere': self.sphere,
         '4neighbor': self.neighbor4,
         '8neighbor': self.neighbor8
        }
        
    def neighbor8(self, d):
        struct = zeros((5, 5), dtype='uint8')
        struct[1:4,:] = 1
        struct[:,1:4] = 1
        if d == 0:
            return struct.astype(bool)
        elif d is not 0:                
            buffer = [struct.copy() for _ in range(d)]
            new_struct = stack(buffer, axis=0)
            return new_struct.astype(bool)
        
    def neighbor4(self, d):
        struct = zeros((3, 3), dtype='uint8')
        struct[:,1] = 1
        struct[1,:] = 1
        if d == 0:
            return struct.astype(bool)
        elif d is not 0:                
            buffer = [struct.copy() for _ in range(d)]
            new_struct = stack(buffer, axis=0)
            return new_struct.astype(bool)
        
    def square(self, w):
        pass
        # struct = np.zeros((w, w), dtype='uint8')
        # x, y = np.indices((w, w))
        # mask = (x - w) * (y - w) <= n**2
        # struct[mask] = 1
        # return struct.astype(np.bool)

    def cube(self, w):
        pass
        # struct = np.zeros((2 * w + 1, 2 * w + 1, 2 * w + 1))
        # x, y, z = np.indices((2 * w + 1, 2 * w + 1, 2 * w + 1))
        # mask = (x - w) *  (y - w) * (z - w) <= w**3
        # struct[mask] = 1
        # return struct.astype(np.bool)

    def circle(self, r):
        struct = zeros((2 * r + 1, 2 * r + 1))
        x, y = indices((2 * r + 1, 2 * r + 1))
        mask = (x - r)**2 + (y - r)**2 <= r**2
        struct[mask] = 1
        return struct.astype(bool)

    def sphere(self, r):
        struct = zeros((2 * r + 1, 2 * r + 1, 2 * r + 1))
        x, y, z = indices((2 * r + 1, 2 * r + 1, 2 * r + 1))
        mask = (x - r)**2 + (y - r)**2 + (z - r)**2 <= r**2
        struct[mask] = 1
        return struct.astype(bool)


def find_largest_component(filled_binary):
    """Find the largest connected component in the binary image.
    This function labels the binary image and finds the summation
    of pixels in the labeled regions. The regions with the max pixels
    are chosen. Method adopted from https://tinyurl.com/y3m4qdg4
    
    filled_binary: Binary image without holes
    """
    # Label the regions in the binary image.
    labeled_im, num_of_regions =  label(filled_binary)
    
    # Sum up all pixels in the regions of the labeled binary.
    sizes = sum(filled_binary, labeled_im, range(num_of_regions + 1))
    
    # Store indicies of the regions with the greatest
    # number of pixels.
    mask = sizes == max(sizes)
    
    largest_regions = mask[labeled_im]
    
    return largest_regions, num_of_regions

########### Masking Routine Begins ############

def mm_mask(im_data, thresh, K1, K2):
    sphere_stel = struct_el()
    strt_el1 = sphere_stel.shape['circle'](K1)
    footprints=generate_binary_structure(2, connectivity=1)

    eroded_im1 = grey_erosion(im_data, footprint=footprints, structure=strt_el1)

    m, n = im_data.shape
    threshold_im1 = ones((m, n), dtype='uint8')
    threshold_im1[eroded_im1 < thresh] = 0


    #### Hole-filling step
    filled_holes = binary_fill_holes(threshold_im1, structure=strt_el1)
    
    # Finding the largest connected component
    # code for this was adopted from ...
    # https://tinyurl.com/y62zqely

    largest_reg1, _ = find_largest_component(filled_holes)

    #### Binary Opening Step ####
    # Binary erosion of a binary union
    union_im = threshold_im1 | largest_reg1
    strt_el2 = sphere_stel.shape['circle'](K2)
    bin_erode = binary_erosion(union_im, structure=strt_el2)

    largest_reg2, _ = find_largest_component(bin_erode)

    # Binary dilation of a binary intersect
    im1 = where(threshold_im1 > 0, 1, 0)
    im2 = where(bin_erode > 0, 1, 0)

    intersect = im1 & im2
    dilate_im2 = binary_dilation(intersect, strt_el1)

    return dilate_im2

def unsharp_mask(img, sigma, w, length=1):
    """Perform unsharp masking on the grayscale image given.
    This function performs the following calculation ...
    Ibar_l = I(1+5w) - 5w x Hbar_l * I
    
    img: grayscale image
    """
    # Convolve 1D filters twice to more efficiently
    # perform the equvalent of a 3 by 3 filter.
    Hbar_x = general_gaussian(M=2*length+1, p=1, sig=sigma)
    Hbar_y = Hbar_x.reshape((1, -1))[0]
    
    smoothed = convolve1d(img, Hbar_x, mode='constant')
    smoothed = convolve1d(smoothed, Hbar_y, mode='constant')
    
    # Perform scaling of the initial image.
    img_plus = img.copy() * (1 + 5 * w)
    
    # Store indicies of the regions with the greatest
    # number of pixels.
    smooth_plus = smoothed * 5 * w
    
    return img_plus - smooth_plus
