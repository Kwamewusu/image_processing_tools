#!/usr/bin/env python

import re, glob
import numpy as np
import nibabel as nib
import pydicom as pydcm

from mm_masking import mm_mask, find_largest_component
from pandas import DataFrame, ExcelWriter


# ## Section for sorting DICOM files

# Function for selecting the NIfTI directories of  different TSLs
def getDirFiles(sl_loc):
    # find global address of NIfTI spin-lock images
    func_data = glob.glob(sl_loc + '/*.nii.gz')

    return func_data
    
# Function for picking DICOM file order
def dcmItemize(convention,text):
    ## Regular expressions for naming convention of scanners ##
    if '3Tsem' in convention:
        pattern = '\w+\.\w+\.\w+\.\d+\.(\d+)\.\d+\.\d+\.\d+\.\d+\.dcm'
    elif '3Tge' in convention:
        pattern = '\w+\.\w+\.\w+\.\d+\.\d+\.\d+\.\d+\-\d+\-(\d+)\-\w+\.dcm'
    elif '7Th' in convention:
        pattern = '\w+\.\w+\.\w+\.\d+\.(\d+)\.\d+\.\d+\-\d+\-\d+\-\w+\.dcm'
    elif '7Ta' in convention:
        pattern = '\-\d+\-(\d+)\-\w+\.dcm'
        
    token = re.search(pattern, text)
    return token.group(1) if token else '&'

def function(x, y):
    return int(x[0]) - int(y[0])

def cmp_to_key(mycmp):
# https://docs.python.org/3/howto/sorting.html#sortinghowto
    'Convert a cmp= function into a key= function'
    class K:
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K

def dcmSort(imOrder,fileList):
    idxNList = zip(imOrder, fileList)
    sortedList = [x for x in idxNList]
    newList = sorted(sortedList, key=cmp_to_key(function))
    return newList

# ## Load files

def sliceSorter(image_data,matSize,vol,sortedFiles):
    if "NumberOfTemporalPositions" in image_data:
        # 0x0021, 0x104f is header element for "Locations in Acquisition"
        if 0x0021104f in image_data:
            sliceQuant = int(image_data[0x0021104f].value)
            numOfImages = int(image_data.ImagesInAcquisition)
            timePoint = 0
            sliceNum = 0
            
            img = np.reshape(pydcm.dcmread(sortedFiles[0]).pixel_array.copy(),
                            (matSize[0],matSize[1]))
            vol[:,:,0,0] = img
            
            for i in range(1,numOfImages):
                img = np.reshape(pydcm.dcmread(sortedFiles[i]).pixel_array.copy(),
                                (matSize[0],matSize[1]))
                vol[:,:,sliceNum,timePoint] = img
                if (i % sliceQuant) == 0:
                    timePoint += 1
                    sliceNum = 0
                else:
                    sliceNum += 1
        else:
            numOfLocs = int(image_data.ImagesInAcquisition)
            numOfSlices = 1
            for i in range(numOfLocs):
                img = pydcm.dcmread(sortedFiles[i]).pixel_array.copy()
                vol[:,:,:,i] =                 np.reshape(img,(matSize[0],matSize[1],numOfSlices)).copy()
    else:
        raise UserWarning("Please make sure this set of DICOMs is functional data")
                
def sliceSorter_nii(image_data,seriesCount,vol,seriesFiles):

    """ Take sorted absolute path and store the image data
    by slice at each spin-lock time after loading NIfTI the image
    """
    # Place images of same series together as a set
    if 'fse' in seqName:
        for i,j in enumerate(seriesFiles):
            img = nib.load(j[0])
            vol[:,:,:,i] = img.get_data()
    elif 'fid' in seqName:
        for i,j in enumerate(seriesFiles):
            img = nib.load(j[0])
            vol[:,:,:,i] = scalar[i] * img.get_data()
    elif 'gre' in seqName:
        for i,j in enumerate(seriesFiles):
            img = nib.load(j[0])
            vol[:,:,:,i] = scalar[i] * img.get_data()
    else:
        raise UserWarning('seqName is must contain "fse","fid" or "gre"')

# ### Interpolation

def interpolate(vol,temporalInterp):
    # Handle temporal interpolation
    if (temporalInterp >= 2):
        sets = timePos
        Nt = timePos * 2
    else :
        sets = np.floor(timePos/tslCount)

    if (temporalInterp == 0):
        return
    elif (temporalInterp == 1):
        # Low spin-lock interpolation
        for sn in range(sets - 1):
            vol[:,:,:,sn*2] = (vol[:,:,:,sn*2] + vol[:,:,:,sn*2+2])/2
    elif (temporalInterp == 2):
        # High spin-lock interpolation
        for sn in range(sets - 1):
            vol[:,:,:,sn*2+1] = (vol[:,:,:,sn*2+1] + vol[:,:,:,sn*2+3]) / 2
    elif (temporalInterp == 3):
        # sliding window interpolation
        volTemp = np.zeros((matrix_size[0], matrix_size[1], numOfSlices, Nt))
        # handle boundary conditions
        volTemp[:,:,:,0:1] = vol[:,:,:,0:1]
        volTemp[:,:,:,sets*2:sets*2+1] = vol[:,:,:,sets:sets+1] 
        for sn in range(1, sets):
            volTemp[:,:,:,sn*2+np.mod(sn,2)] = vol[:,:,:,sn]
            volTemp[:,:,:,sn*2+1-np.mod(sn,2)] = vol[:,:,:,sn-1]
        vol = volTemp
        del volTemp
    elif (temporalInterp == 4):
        # sliding window interpolation + temporal interpolation
        volTemp = np.zeros((matrix_size[0], matrix_size[1], numOfSlices, Nt))
        # handle boundary conditions
        volTemp[:,:,:,0:1] = vol[:,:,:,0:1]
        volTemp[:,:,:,sets*2:sets*2+1] = vol[:,:,:,sets:sets+1] 
        for sn in range(1, sets):
            volTemp[:,:,:,sn*2+np.mod(sn,2)] = vol[:,:,:,sn]
            volTemp[:,:,:,sn*2+1-np.mod(sn,2)] = (vol[:,:,:,sn-1] + vol[:,:,:,sn+1]) / 2
        vol = volTemp
        del volTemp

# ## Save sorted spin-lock images

# DICOM to NIfTI conversion info
def dcm2niConvert(refIm, zoomInfo, imToSave, saveAs):
    # Make F array from DICOM orientation page
    F = np.fliplr(np.reshape(refIm.ImageOrientationPatient, (2, 3)).T)
    rotations = np.eye(3)
    rotations[:, :2] = F

    # Third direction cosine from cross-product of first two
    rotations[:, 2] = np.cross(F[:, 0], F[:, 1])

    # Add the zooms
    zooms = np.diag(zoomInfo)

    # Make the affine
    affine = np.diag([0., 0, 0, 1])    
    affine[:3, :3] = rotations.dot(zooms)    
    affine[:3, 3] = refIm.ImagePositionPatient

    # Make NIfTI image object and save it
    tslIm = nib.Nifti1Image(imToSave, affine)
    nib.save(tslIm, saveAs)


# Function for saving NIfTI images
def save_as_nii(affine, head, imToSave, saveAs):
    ''' The affine matrix porduced by this function was taken from
    https://nipy.org/nibabel/dicom/dicom_orientation.html#dicom-slice-affine
    '''
    # Make NIfTI image object and save it
    tslIm = nib.Nifti1Image(imToSave, affine)
    tslIm.header.set_qform(head.get_qform())

    oldSform = head.get_sform(coded=True)
    tslIm.header.set_sform(oldSform[0],oldSform[1])

    tslIm.header.set_data_shape(head.get_data_shape())
    nib.save(tslIm, saveAs)


# ### Functions for calculating T1r and masking 
# Function with multiple mask options
def maskProg(unMasked,thresh,case):
    
    # Get dimensions of unmasked image
    m, n, numOfSlices = unMasked.shape
    
    # Create placeholders of averaging operation & final mask volume

    mask = np.zeros((numOfSlices,m, n))
    
    if case == 0:
    # Perform averaging of each slice/loc
        meanVol = np.zeros((numOfSlices,m, n))
        for l in range(numOfSlices):
            avgThruT = np.mean(unMasked[l,:,:,0], axis=0)
            avgVal = [mu for mu in avgThruT]
            meanVol[l,:,:] = avgVal

        mask[meanVol > thresh] = 1

        largest_region, _ = find_largest_component(mask)
        
        del m, n, numOfSlices, mask
        return largest_region

    elif case == 1:
    # Create mask from median values
        tempBuffer = np.zeros((m,n))
        for l in range(numOfSlices):
            maskBuffer = np.squeeze(unMasked[l,:,:,0])
            tempBuffer[maskBuffer > 2.5*np.median(maskBuffer.ravel('F'))] = 1
            mask[l,:,:] = tempBuffer

        del m, n, numOfSlices, tempBuffer, maskBuffer
        return mask

    elif case == 2:
    # Create mask from mathematical morphology
        for l in range(numOfSlices):
            maskBuffer = mm_mask(unMasked[l,:,:,0], thresh, 2, 1)
            mask[l,:,:] = maskBuffer

        del m, n, numOfSlices, maskBuffer
        return mask

def productFunc(mat1,mat2):
    
    # Multiply two arrays element-wise
    prod = np.multiply(mat1, mat2)
    return prod


# Function for applying a mask
def applyMask(img, mask2Use):
    dims1 = np.ndim(img)
    dims2 = np.ndim(np.squeeze(mask2Use))
    
    if dims1 > dims2:
        volProd = np.zeros((img.shape))

        doProduct = np.vectorize(productFunc,otypes=[np.float],cache=False)
        for idx in range(img.shape[3]):
            volProd[:,:,:,idx] = doProduct(img[:,:,:,idx], mask2Use)
    else:
        volProd = np.multiply(img, mask2Use)
    
    return volProd


# Function for T1rho estimation
def t1rhoCalc(non0Len, indices, TSLs, tslIms, t1rVolBuffer, twoTSLs):

    # Special case where only 2 spin-lock times are acquired
    if twoTSLs == 1:

        # Special case where only 2 spin-lock times are acquired
        sigOut = np.array(np.zeros((2,non0Len)))
        
        # Numerator, diff. between max and min spin-lock times
        dTSL = TSLs[1] - TSLs[0]

        """ Fill min and max vectors separately.
        The 'A' option tells Python not to change the index
        ordering of the array being operated on.
        """
        sigOut[0,:] = tslIms[:,:,:,0].ravel('A').take(indices)
        sigOut[1,:] = tslIms[:,:,:,1].ravel('A').take(indices)

        # Compute the slope (T1rho); T1r = -t/log(Stsl/S0)
        t1rVolBuffer.ravel(order='A')[indices] = \
-dTSL/np.log(np.true_divide(sigOut[1,:],sigOut[0,:],order='A'),order='A')
    else:

        # Case for spin-lock times > 2
        for ni in range(non0Len):
            for i in range(len(TSLs)):
                sigOut[i,:] = tslIms[:,:,:,i].ravel('A').take(indices)
            P = np.polyfit(TSLs, np.log(sigOut), 1)
            t1rVolBuffer.ravel(order='A')[indices[ni]] = -1/P[0]
    return t1rVolBuffer


def mean_wrt_time(img, mask, out_list):
    """Use the mask provided to extract the relevant
    regions in the image given. Compute the mean for
    each time point.

    img:ndarray: Single slice image of 4 dimensions
    mask:ndarray: Image of 2 dimensions
    out_list:list: Python iterable for storage of computed means
    """

    m, n, s, timepoints = img.shape

    for t in range(timepoints):
        buffer = img[:,:,s,t].reshape((m,n,s))
        region_vals = buffer[mask]

        out_list.append( np.mean(region_vals))

def save_as_excel(out_name, df_iterable, name_list):
    """Save the list of Pandas DataFrame objects with
    their respective sheet names.
    
    out_name:str: Address/filename for file to be saved
    df_iterable:list: Iterable containing pandas DataFrame objects
    name_list:list: Iterable of sheet names
    """

    with ExcelWriter(out_name) as writer:
        for idx, df in enumerate(df_iterable):
            df.to_excel(writer, sheet_name=list_name[idx])
