import math
import numpy as np
import pandas as pd
import pathlib
import astropi.io import fits

import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
import matplotlib.cm as mpl_cm
import seaborn as sns
sns.set(context='notebook',style='ticks',font_scale=1.5,palette='bright')


#Constants for data reduction
sol = 299792458 #m/s
planck_joule = 6.6267015e-34 #Joule s
elemcharge =  1.602176634e-19 #coulombs
planck = planck_joule / elemcharge #eV s
metertoang = 10**(10)

def check_ROI(file,
              h=np.ndarray([15]),
              w=np.ndarray([15]),
              edge=(5,5),
              mask=None,
              darkside='LHS',
              diz_threshold=10,
              diz_size=3
             ):
    """
    Parameters
    ----------
        file : path to .fits file 
            .fits file to load and process
        h / w : int (np.ndarray
            List of height and width values to calculate the specular condition.
        edge : tuple 
            number of pixels to trim from the edge of the image before processing.            
        mask : np.ndarray, Boolean
            Array of pixels to ignore during calculation. Only pixels set to `True` will be considered.
        darkside : 'LHS' or 'RHS'
            Side of the image to take the dark frame.
        diz_threshold / diz_size : int
            Dizinger properties to remove 'hot' pixels.
        d : Boolean 
            Display a summary plot
    """
    signal = np.zeros([len(h), len(w)])
    #Grab information
    meta = load_fits(file)
    image = meta['image']
    
    for i, height in enumerate(h):
        for j, width in enumerate(w): 
            image_spot, _, beamspot, _ = slice_spot(image,
                                                    (round(height)),
                                                    (round(width)),
                                                    mask=mask,
                                                    rect=True,
                                                    edge=edge,
                                                    diz_threshold=diz_threshold,
                                                    diz_size=diz_size
                                                    )
            image_dark, _ = slice_dark(image,
                                       (round(height)),
                                       (round(width)),
                                       beamspot=beamspot,
                                       rect=True,
                                       darkside=darkside
                                      )
            refl = int(image_spot.sum())
            dark = int(image_dark.sum())
            signal[i,j] = refl-dark
        
    return signal

def check_spot(file,
               h=18,
               w=15,
               edge=(5,5),
               mask=None,
               darkside='LHS',
               diz_threshold=10,
               diz_size=3,
               d=True
              ): 
    """
    Parameters
    ----------
        file : path to .fits file 
            .fits file to load and process
        h / w : int (even number)
            height and width of ROI used to calculate reflectivity. Center is defined as pixel with hightest magnitude.
        edge : tuple 
            number of pixels to trim from the edge of the image before processing.            
        mask : np.ndarray, Boolean
            Array of pixels to ignore during calculation. Only pixels set to `True` will be considered.
        darkside : 'LHS' or 'RHS'
            Side of the image to take the dark frame.
        diz_threshold / diz_size : int
            Dizinger properties to remove 'hot' pixels.
        d : Boolean 
            Display a summary plot

    """
    #Grab important information
    meta = load_fits(file)
    energy = meta['energy']
    angle_theta = meta['angle_theta']
    image = meta['image']
            
    #Calculate q-value        
    lam = metertoang*planck*sol/(energy) #calculate wavelength
    qval = 4 * np.pi * np.sin(angle_theta*np.pi/180)/lam #calculate q-values
    
    #Process image
    image_spot, image_avg, beamspot, rect_spot = slice_spot(image, h, w,
                                                            mask=mask,
                                                            rect=True,
                                                            edge=edge,
                                                            diz_threshold=diz_threshold,
                                                            diz_size=diz_size
                                                            )
    image_dark, rect_dark = slice_dark(image, h, w, beamspot=beamspot, rect=True, darkside=darkside)
    
    I_refl = int(image_spot.sum())
    dark = int(image_dark.sum())
    SNR = float((I_refl/dark)) ### "signal to noise ratio"
    
    #Relevant outputs that you would want to read.
    print('File: {}'.format(file))
    for key in meta:
        if key=='image':
            continue
        print(key+': {}'.format(meta[key]))

    print('Processed Variables:')
    print('Q:', qval)
    print('Specular:',I_refl)
    print('Background:',dark)
    print('Signal:', I_refl - dark)
    print('SNR:', SNR)
    print('Beam center', beamspot)

    if d==True:
        dx=edge[0]
        dy=edge[1]
        fig, ax = plt.subplots(1,4,subplot_kw={'xticks':[],'yticks':[]},figsize=(12,12))
        ax[0].imshow(image[dx:-dx,dy:-dy],norm=mpl_colors.LogNorm(),cmap='terrain')
        ax[1].imshow(image_avg,norm=mpl_colors.LogNorm(),cmap='terrain')
        
        if mask is not None:
            mask_display = np.ma.masked_where(mask == True, mask)
            ax[0].imshow(mask_display[dx:-dx,dy:-dy],cmap='Greys_r')
            ax[1].imshow(mask_display[dx:-dx,dy:-dy],cmap='Greys_r')
        ax[0].add_patch(rect_spot)
        ax[0].add_patch(rect_dark)
            
        ax[2].imshow(image_spot,norm=mpl_colors.LogNorm(),cmap='terrain')
        ax[3].imshow(image_dark,norm=mpl_colors.LogNorm(),cmap='terrain')
        plt.show()
            
    return image, image_avg, image_spot, image_dark



def slice_spot(image,
               h=20,
               w=20,
               edge=(5,5),
               mask=None,
               rect=False,
               diz_threshold=1.5,
               diz_size=3
              ): 
    """
    Parameters
    ----------
        image : np.ndarray 
            Image to process.
        h / w : int (even number)
            height and width of ROI used to calculate reflectivity. Center is defined as pixel with hightest magnitude. 
        edge : tuple 
            number of pixels to trim from the edge of the image before processing.
        mask : np.ndarray, Boolean
            Array of pixels to ignore during calculation. Only pixels set to `True` will be considered.
        rect : boolean
            Will export location of the average rectangle for visualization.  
        diz_threshold / diz_size : int
            Dizinger properties to remove 'hot' pixels.
    """
    dx=edge[0]
    dy=edge[1]
    
    #Check if mask is an appropriate input, if not, generate dummy mask and ignore.
    if mask is None:
        mask = np.full(image.shape, True) #No Mask
    elif mask.shape != image.shape:
        mask = np.full(image.shape, True)
    
    image_zinged, image_locbeam = DezingerImage(image[dx:-dx,dy:-dy], threshold=diz_threshold, size=diz_size)
    
    mask_trim = mask[dx:-dx,dy:-dy]# Cut the edge off the mask
    image_zinged[~mask_trim]=0 #Use Boolean Indexing to set all 'False' pixel to 0 for the sake of finding the image maximum
    image_locbeam[~mask_trim]=0
    
    #y_spot_test,x_spot_test=np.unravel_index(np.argmax(IMAGE_MASKED),IMAGE_MASKED.shape) #Find the highest pixel
    y_spot,x_spot= np.unravel_index(np.argmax(image_locbeam),image_locbeam.shape) #Find the highest pixel

    y_low_bound = y_spot - (h//2)#+dy Already include trim in image_zinged
    x_low_bound = x_spot - (w//2)#+dx Already include trim in image_zinged
   
    sl1 = slice(y_low_bound,y_low_bound+h)
    sl2 = slice(x_low_bound,x_low_bound+w)
    sl_spot = (sl1,sl2)
    image_out = image_zinged[sl_spot]
    
    rect_spot = plt.Rectangle((x_low_bound, y_low_bound),w,h,edgecolor='green',facecolor='None')
    
    return image_out, image_locbeam, (x_spot, y_spot), rect_spot

    
def slice_dark(image,
               h=20,
               w=20,
               edge=(5,5),
               mask=None,
               rect=False,
               beamspot=(75,75),
               darkside='LHS'
              ): 
    """
    Parameters
    ----------
        image : np.ndarray 
            Image to process.
        h / w : int (even number)
            height and width of ROI used to calculate reflectivity. Center is defined as pixel with hightest magnitude.
        beam_spot : tuple
            Location of the beam as determined by slice_spot
        edge : tuple 
            number of pixels to trim from the edge of the image before processing.
        mask : np.ndarray, Boolean
            Array of pixels to ignore during calculation. Only pixels set to `True` will be considered.
        rect : boolean
            Will export location of the average rectangle for visualization.
        darkside ('LHS' or 'RHS')
            Side of the image to take the dark frame.

    """  
    dx=edge[0]
    dy=edge[1]
    
    #Check if mask is an appropriate input, if not, generate dummy mask and ignore.
    if mask is None:
        mask = np.full(image.shape, True) #No Mask
    elif mask.shape != image.shape:
        mask = np.full(image.shape, True)

    y_spot = beamspot[1]
    if darkside == 'LHS':
        x_dark =  edge[0]  ### box on left side of image
    elif darkside == 'RHS':
        x_dark =  image.shape[0]-edge[0]-w  ### box on right side of image
    else: #Default to RHS
        x_dark =  image.shape[0]-edge[0]-w  ### box on right side of image
    
    y_dark =  y_spot-(h//2)#+dy Already include trim in image_zinged
        
    sl1 = slice(y_dark,y_dark+h)
    sl2 = slice(x_dark,x_dark+w)
    sl_dark = (sl1,sl2)
    image_dark = image[sl_dark]
    
    rect_dark = plt.Rectangle((x_dark,y_dark),w,h,edgecolor='red',facecolor='None')
    return image_dark, rect_dark

    
    
    
    
    
    
    
    
    
"""
UTILITY FUNCTIONS FOR THE ABOVE
"""
def load_fits(file):
    out = {}
    with fits.open(file) as hdul:
            out['exposure'] = hdul[0].header['EXPOSURE']
            out['beam_current'] = hdul[0].header['HIERARCH Beam Current']
            out['angle_theta'] = hdul[0].header['Sample Theta']
            out['theta2theta'] = hdul[0].header['T-2T']
            out['ccd_theta'] = hdul[0].header['CCD Theta']
            out['image'] = hdul[2].data
            out['energy'] = round(hdul[0].header['HIERARCH Beamline Energy'],2) #eV
            out['polarization'] = hdul[0].header['EPU POLARIZATION']
            out['hos'] = round(hdul[0].header['HIERARCH Higher Order Suppressor'],1)
            out['hes'] = hdul[0].header['HIERARCH Horizontal Exit Slit Size']
    return out
    

#Replace pixels above a threshold with the average defined by a box of SIZE x SIZE around the pixel
#Also returns the averaged display (based on size) for use in defining where the beam center is
def DezingerImage(image, threshold=1.5, size=3):
    from scipy import ndimage
    med_result = ndimage.median_filter(image, size=size) #Apply Median Filter to image
    diff_image = image / np.abs(med_result) #Calculate Ratio of each pixel to compared to a threshold
    #Repopulate image by removing pixels that exceed the threshold -- From Jan Ilavsky's IGOR implementation.
    output = image*np.greater(threshold, diff_image).astype(int) + med_result*np.greater(diff_image, threshold) #
    return output, med_result #Return dezingered image and averaged image