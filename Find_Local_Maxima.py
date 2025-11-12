
# ================ DESCRIPTION ==============================================================================
#
# This file reunites the different functions used to determine the positions of blobs in fluorescent images as
# well as the position of local maxima in theses blobs. The functions are then applied in our Detection_algorithm.py file
# to retrieve the positions of the cells in each image.
#
# =============== REQUIRED PACKAGES =========================================================================================

import numpy as np # to use arrays
import cv2 # for image treatment
import pandas # to use dataframes
from scipy import ndimage as ndi # for multidimensionnal image processing
from scipy.spatial import KDTree

# =============== FUNCTIONS =========================================================================================

# Function that applies the dilation and erosion method on an image to 
# determine the local maxima
# Arguments:
#   - img: image as matrix of pixels
# Output:
#   - dst: matrix of pixels with 1 where local maxima have been detected and 0 elsewhere
def localMax(img):
    kernel = np.ones((3,3), np.uint8) 
    img_dilation = cv2.dilate(img, kernel, iterations=1) 
    img_max=cv2.compare(img,img_dilation,cv2.CMP_GE)
    img_erode=cv2.erode(img, kernel, iterations=1) 
    img_min=cv2.compare(img,img_erode,cv2.CMP_GT)
    # retrieve pixels where max by dilation are also minimums by erosion
    dst=cv2.bitwise_and(img_max,img_min)
    return dst


# Function that 
# Arguments:
#   - blobs: dataframe of pixels coordinates (columns "x" and "y") and their value (column "value")
#   - shape: shape of our images
# Output:
#   - p: dataframe of two columns "x" and "y" corresponding to the positions of the cells
def findMax(blobs,shape):
    # we define a nbew image with the pixel values of the blobs on it and zero elsewhere
    new_img=np.zeros(shape)
    new_img[blobs.iy,blobs.ix]=blobs.val
    #  we find the local maximas
    locmax=localMax(new_img)
    # we remove neighboring max
    Nc, markers = cv2.connectedComponents(locmax)
    iy,ix=np.where(markers>0)
    p=pandas.DataFrame(np.array([ix,iy,markers[iy,ix]]).T,columns=["x","y","mark"])
    # we define ƒmean maxes
    p=p.groupby('mark').agg({'x':'mean','y':'mean'})
    return p[['x','y']]

# Function that first applies a smoothing before using the Laplacian to 
#  identify contours
# Arguments:
#   - yy: image as matrix of pixels
#   - sigma: value of the smoothing to apply
# Output:
#   - lap: matrix of pixels with negative values where contours have been detected
def LoG(yy,sigma):
    s1=ndi.gaussian_filter(yy,sigma)
    lap=-ndi.laplace(s1)
    return lap

# Function that 
# Arguments:
#   - yy: image as matrix of pixels
#   - s: value of the threshold applied to determine the interior of the contours
#   - ccmin: maximum number of pixels for a blob to be processed (default value to 20)
#   - ccmax: maximum number of pixels for a blob to be processed (default value to None)
#   - sigma: value of the smoothing (default value to 2)
#   - method: method used to identify blobs in the pictures (default to "LoG")
#   - returnMap: boolean to specify if one wish to also have the result of the LoG in output
# Output:
#   - q: dataframe of 3 columns "ix", "iy" and "val" corresponding to the pixels 
def getBlobs(yy,s,ccmin=20,ccmax=None,sigma=2,method="LoG",returnMap=False):
    # First, apply smoothing and Laplacian through LoG function defined previously
    if method=="LoG":
        lap=LoG(yy,sigma)
    else:
        assert False,"{} unknown method".format(method)
    # Apply threshold
    y_t=np.where(lap>s,lap,s)
    img_t = cv2.normalize(y_t, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # Use connected components to retrieve and label blobs
    Nc, markers = cv2.connectedComponents(img_t)
    # Check the criteria of minimum size
    iy,ix=np.where(markers>0)
    d={'ix':ix,'iy':iy,'mark':markers[iy,ix],'val':y_t[iy,ix]}
    # create a dataframe from a dictionnary
    q=pandas.DataFrame(d)
    # this way we can retrieve the connected components that respect the criteria
    cnt=q.groupby("mark").size().to_frame("sz").reset_index()
    cnt=cnt[cnt.sz>=ccmin]
    # check the criteria of maximum size if not None
    if ccmax is not None:
        cnt=cnt[cnt.sz<ccmax]
    q=q.merge(cnt,on='mark')  
    q=q[['ix','iy','val']]
    if returnMap:
        return q,lap
    else:
        return q
    

def filter_coordinates(pos, distance_min):
    if pos.shape[0] == 0:
        return pos  # If there are no coordinates, return the empty array
    pos = np.array(pos, dtype=float)
    # Create a KDTree for efficient spatial queries
    tree = KDTree(pos)
    
    # Initialize an empty list to store filtered coordinates
    filtered_pos = []
    
    # Iterate over each point in pos
    for i, point in enumerate(pos):
        # Query points within distance_min (excluding the point itself)
        indices = tree.query_ball_point(point, distance_min)
        # Filter out points that are already included in filtered_pos
        is_close_to_filtered = any(
            np.array_equal(pos[idx], existing_point) for idx in indices for existing_point in filtered_pos
        )
        if not is_close_to_filtered:
            filtered_pos.append(point)
    return np.array(filtered_pos)
