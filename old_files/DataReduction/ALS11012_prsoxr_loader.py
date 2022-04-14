"""
Some amount of copyright:

"""
import math
import numpy as np
import pandas as pd
import pathlib
from astropy.io import fits
import h5py
from ALS11012_prsoxr_reduction import load_fits, DezingerImage

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

class ALS_PRSoXR_Loader():
    """
    Class to load PRSoXR data from beamline 11.0.1.2 at the ALS
    
    Parameters
    -----------
    files : list
        List of .fits to be loaded. Include full filepaths
        
        >>> #Recommended Usage
        >>> import pathlib
        >>> path_s = pathlib.Path('../ALS/2020 Nov/MF114A/spol/250eV')
        >>> files = list(path_s.glob('*fits')) # All .fits in path_s
        
    name : str
        Name associated with the dataset. Will be used when saving data.
        
    mask : np.ndarray (Boolean)
        Array of equal dimensions to an individual image. Elements set to `False` will be excluded when finding beamcenter.
        
    autoload : Boolean
        Set to false if you do not want to load the data upon creation of object.
        
    Attributes
    -----------
    name : str
        Human readable string that describes the dataset to be loaded. See 'name' parameter
    mask : np.ndarray (Bool)
        Data mask to be applied to all images. 
    files : list
        List of filepaths
    shutter_offset : float
        Piezo shutter deadtime to be added to all exposures. It is recommended to measure at the time of taking data (see online tutorial)
    sample_location : int
        Sample location on the holder: Bottom of holder == 180, Top of holder == 0. Will be automatically updated when files load
    angle_offset : float
        Angle offset [deg] to be added to 'Sample Theta' to correctly calculate q. (see online tutorial)
    energy_offset : float
        Energy offset [eV] to be applied to 'Beamline Energy' to correctly calculate q.
    snr_cutoff : float
        Ratio of dark counts vs. light counts. Any image below this threshold will be excluded from processing.
    variable_motors : list(str)
        List of upstream optics that were varied to modify upstream flux.
    imagex : int
        X-dimension of .fits. Will be automatically updated when files load
    imagey : int
        Y-dimension of .fits. Will be automatically updated when files load
    edge_trim : tuple(int)
        Edge of the detector that you want to ignore in processing. (x dim, y dim)
    darkside : 'LHS' or 'RHS'
        Side of the detector that you want to calculate the dark counts.
    diz_threshold : int
        Dizinger intensity threshold to remove 'hot' pixels.
    diz_size : int
        Size of box to average to remove 'hot' pixels.
        
    Notes
    ------
    
    Print the loader to view variables that will be used in reduction. Update them using the attributes listed in this API.
    
    >>> loader = ALS_PRSoXR_Loader(files, name='MF114A_spol')
    >>> print(loader) #Default values
        Sample Name - MF114A
        Number of scans - 402
        ______________________________
        Reduction Variables
        ______________________________
        Shutter offset = 0.00389278
        Sample Location = 0
        Angle Offset = -0.0
        Energy Offset = 0
        SNR Cutoff = 1.01
        ______________________________
        Image Processing
        ______________________________
        Image X axis = 200
        Image Y axis = 200
        Image Edge Trim = (5, 5)
        Dark Calc Location = LHS
        Dizinger Threshold = 10
        Dizinger Size = 3
    >>>loader.shutter_offset = 0.004 #Update the shutter offset
    >>>
        
    Once process attributes have been setup by the user, the function can be called to load the data. An ROI will need to be specified at the time of processing. Use the ``self.check_spot()`` function to find appropriate dimensions.
    
    >>> refl = loader(h=40, w=30)
    
    Data that has been loaded can be exported using the ``self.save_csv(path)`` and ``self.save_hdf5(path)`` functions.
    
    """
    
    def __init__(self, files, name=None, mask=None, autoload=True):
    
        self.files = files # List of files to be loaded
        self.mask = mask # Mask to be applied to all images
        self.name = name # Name of the series to be loaded

        self.shutter_offset = 0.00389278 # [s]
        self.sample_location = 0 # sample on bottom of holder = 180, sample on top = 0
        self.angle_offset = 0 # [deg]
        self.energy_offset = 0 # [eV]
        self.percent_error = 1.5 # [%] Uncertainty option (legacy option)
        self.snr_cutoff = 1.01 # SNR = I_refl/dark ~ Cutoff on if frame is ignored.
        self.variable_motors = ['Higher Order Suppressor', 'Horizontal Exit Slit Size']
        
        # Information for Image processing
        self.imagex = 200
        self.imagey = 200
        self.edge_trim = (5,5)
        self.darkside = 'LHS'
        
        # Dezinger image variables (If gamma rays become a problem)
        self.diz_threshold = 10
        self.diz_size = 3
        
        # Build storage lists
        self.images = []
        self.image_data = None
        self.normalized_data = None

        self.meta = None
        self.refl = None
        
        # Stats that will be calculated
        self.I0 = None
        self.I0_err = None
        self._scale_factors = None
        self._stitch_points = None 
        
        if autoload == True:
            self.images, self.meta = load_prsoxr_fits(self.files)
            _ = self._update_stats(10)        
        
    def __str__(self):
    
        s = []#["{:_>50}".format("")]
        s.append("Sample Name - {0}".format(self.name))
        s.append("Number of scans - {0}".format(len(self.files)))
        s.append("{:_>30}".format(""))
        s.append("Reduction Variables")
        s.append("{:_>30}".format(""))
        s.append("Shutter offset = {0}".format(self.shutter_offset))
        s.append("Sample Location = {0}".format(self.samp_loc))
        s.append("Angle Offset = {0}".format(self.angle_offset))
        s.append("Energy Offset = {0}".format(self.energy_offset))
        s.append("SNR Cutoff = {0}".format(self.snr_cutoff))
        s.append("{:_>30}".format(""))
        s.append("Image Processing")
        s.append("{:_>30}".format(""))
        s.append("Image X axis = {0}".format(self.imagex))
        s.append("Image Y axis = {0}".format(self.imagey))      
        s.append("Image Edge Trim = {0}".format(self.edge_trim))
        s.append("Dark Calc Location = {0}".format(self.darkside))
        s.append("Dizinger Threshold = {0}".format(self.diz_threshold))
        s.append("Dizinger Size = {0}".format(self.diz_size))
        
        return "\n".join(s)

    def __call__(self, h=25, w=25):
        """
        
        """
        refl = self._calc_refl(h, w)
        return refl
        
        
        
    def check_spot(self, file,
                   h=18,
                   w=15,
                   d=True
                  ): 
        """
        Function to quickly load, reduce, and display a single frame. 
        
        Parameters
        ----------
            file : int 
                Frame index to view. Extracts data from self.image and self.meta
                
            h : int (even number)
                height of ROI used to calculate reflectivity. Center is defined as pixel with hightest magnitude.
            
            w : int (even number)
                width of ROI used to calculate reflectivity. Center is defined as pixel with hightest magnitude.
            edge : tuple 
                number of pixels to trim from the edge of the image before processing.
                
            mask : np.ndarray, Boolean
                Array of pixels to ignore while locating the beamspot. Only pixels set to `True` will be considered.
                
            darkside : 'LHS' or 'RHS'
                Side of the image to take the dark frame.
                
            diz_threshold : int
                Dizinger intensity threshold to remove 'hot' pixels.
                
            diz_size : int
                Size of box to average to remove 'hot' pixels.
                
            d : Boolean 
                Display a summary plot
                
        Returns
        --------
            processed images : list
                Arrays of the following images: Raw Image, Median Filter, Beam spot, dark frame
                
        Notes
        ------
        Will output general motor positions about the frame in question.
        
        If d==1, the images will be displayed according to the output. Black rectangle represents the beam spot. Red rectangle represents the dark frame.
        
        >>> #Quickly verify image stats
        >>> frame = loader.check_spot(file=11, h=40, w=20)
            Exposure: 0.00100000004749745
            Beam Current: 0.0
            Angle Theta: 1.112
            T-2T: 1.112
            CCD Theta: 2.221
            Photon Energy: 249.996035379532
            Polarization: 100.0
            Higher Order Suppressor: 11.9997548899767
            Horizontal Exit Slit Size: 100.0
            Processed Variables:
            Q: 0.004916964189179494
            Specular: 845685
            Background: 115456
            Signal: 730229
            SNR: 7.32473842849224
            Beam center (85, 51)
        >>>
        
        """
        
        #Load frame if it is not yet loaded:
        if len(self.images) == 0:
            image, meta = load_prsoxr_fits(self.files[file])
        else:
            image = self.images[file]
            meta = self.meta.iloc[file]
            
        energy = round(meta['Beamline Energy'], 1)
        angle_theta = meta['T-2T']

        #Calculate q-value        
        lam = metertoang*planck*sol/(energy) #calculate wavelength
        qval = 4 * np.pi * np.sin(angle_theta*np.pi/180)/lam #calculate q-values
        
        #Process frame
        image_spot, image_avg, beamspot, rect_spot = slice_spot(image, h, w,
                                                                mask=self.mask, edge_trim=self.edge_trim,
                                                                diz_threshold=self.diz_threshold,
                                                                diz_size=self.diz_size
                                                                )
        image_dark, rect_dark = slice_dark(image, h, w, beamspot=beamspot,
                                           mask = self.mask, edge_trim=self.edge_trim,
                                           darkside=self.darkside
                                           )
        
        I_refl = int(image_spot.sum())
        dark = int(image_dark.sum())
        SNR = float((I_refl/dark)) ### "signal to noise ratio"
        
        #Relevant outputs that you would want to read.
        print('Exposure: {}'.format(meta['EXPOSURE']))
        print('Beam Current: {}'.format(meta['Beam Current']))
        print('Angle Theta: {}'.format(meta['Sample Theta']))
        print('T-2T: {}'.format(meta['T-2T']))
        print('CCD Theta: {}'.format(meta['CCD Theta']))
        print('Photon Energy: {}'.format(meta['Beamline Energy']))
        print('Polarization: {}'.format(meta['EPU Polarization']))
        print('Higher Order Suppressor: {}'.format(meta['Higher Order Suppressor']))
        print('Horizontal Exit Slit Size: {}'.format(meta['Horizontal Exit Slit Size']))
        print('\n')
        print('Processed Variables:')
        print('Q:', qval)
        print('Specular:',I_refl)
        print('Background:',dark)
        print('Signal:', I_refl - dark)
        print('SNR:', SNR)
        print('Beam center', beamspot)

        if d==True:
            dx=self.edge_trim[0]
            dy=self.edge_trim[1]
            fig, ax = plt.subplots(1,4,subplot_kw={'xticks':[],'yticks':[]},figsize=(12,12))
            ax[0].imshow(image[dx:-dx,dy:-dy],norm=mpl_colors.LogNorm(),cmap='terrain')
            ax[1].imshow(image_avg,norm=mpl_colors.LogNorm(),cmap='terrain')
            
            if self.mask is not None:
                mask_display = np.ma.masked_where(self.mask == True, self.mask)
                ax[0].imshow(mask_display[dx:-dx,dy:-dy],cmap='Greys_r')
                ax[1].imshow(mask_display[dx:-dx,dy:-dy],cmap='Greys_r')
            ax[0].add_patch(rect_spot)
            ax[0].add_patch(rect_dark)
                
            ax[2].imshow(image_spot,norm=mpl_colors.LogNorm(),cmap='terrain')
            ax[3].imshow(image_dark,norm=mpl_colors.LogNorm(),cmap='terrain')
            plt.show()
                
        return [image, image_avg, image_spot, image_dark]
        
    def _calc_refl(self, h=25, w=25, tol=2):
        """
        Function that performs a complete data reduction of prsoxr data 
        """
        self._reduce_2D_images(h=h, w=w, tol=2)
        self._normalize_data()
        self._find_stitch_points()
        self._calc_scale_factors()
        self.refl = self._stitch_refl()
        
        return self.refl                
                
    def _reduce_2D_images(self, h=25, w=25, tol=2):
        """
        
        Internal function that calculates the reduced specular reflectivity of all files within ``self.files `` 

        """
        data = []
        for i, image in enumerate(self.images):
            _vars = self.meta.iloc[i]
            wavelength = metertoang*planck*sol/round(_vars['Beamline Energy'] + self.energy_offset, 1)
            if np.round(_vars['CCD Theta'],tol) == 0: #Check if its the direct beam
                q = 0
            else:
                q = 4 * np.pi *np.sin(np.radians((_vars['Sample Theta'] - self.angle_offset)))/wavelength
                
            exposure = _vars['EXPOSURE'] + self.shutter_offset
            beamcurrent = _vars['Beam Current'] if _vars['Beam Current'] > 100 else 1 #Ignore beam current if not measured
            
            image_spot, _, beamspot, _ = slice_spot(image, h, w,
                                                    mask=self.mask, edge_trim=self.edge_trim,
                                                    diz_threshold=self.diz_threshold,
                                                    diz_size=self.diz_size
                                                    )
            image_dark, _ = slice_dark(image, h, w,
                                       mask = self.mask, edge_trim=self.edge_trim,
                                       darkside=self.darkside, beamspot=beamspot
                                       )
            
            i_tot = int(image_spot.sum())
            i_dark = int(image_dark.sum())
            
            snr = float(i_tot/i_dark)
            if snr < self.snr_cutoff or snr < 0:
                continue
            i_refl = (i_tot - i_dark)
            r = ((i_refl)) / (exposure*beamcurrent)
            r_err = np.sqrt(i_tot + i_dark)/(exposure*beamcurrent)
            
            data.append([i, q, r, r_err])
        
        self.image_data = pd.DataFrame(data, columns=(['index', 'Q', 'R', 'R_err']))
        
    def _normalize_data(self):
        """
        
        Internal function that normalizes ``self.image_data`` to the direct beam and updates ``self.normalized_data``
        
        """
        refl = pd.concat([self.image_data, self.meta[self.variable_motors].round(1)], axis=1)
        
        I0_cutoff = refl['Q'].where(refl['Q'] == 0).count()
        if I0_cutoff > 0:
            I0 = refl['R'].iloc[:I0_cutoff].mean()
            I0_err = refl['R'].iloc[:I0_cutoff].std()
        else:
            I0 = 1
            I0_err = 0
        
        self.I0 = I0
        self.I0_err = I0_err
        
        refl['R_err'] = np.sqrt((refl['R']/I0)**2 * ((refl['R_err']/refl['R'])**2 + (I0_err/I0)**2))
        refl['R']=refl['R']/I0
        
        self.normalized_data = refl.drop(refl.index[:I0_cutoff])


    def _find_stitch_points(self):
        """
        Internal function that locates the frames that one of the ``self.variable_motors`` has been changed.

        """
        df = self.normalized_data.drop(['index', 'Q', 'R', 'R_err'], axis=1) # Use legacy code
        idx = []
        imotor = []
        skip = False
        skipCount = 0
        skipCountReset = 2
        for motor in df.columns:
            for i,val in enumerate(np.diff(df[motor])):
                if skip:
                    if (skipCount<=skipCountReset):
                        skipCount+=1
                    else:
                        skip=False
                        skipCount = 0
                elif abs(val)>1e-5:
                    idx.append(i)
                    imotor.append(motor)
                    skip = True
        dfx = pd.DataFrame([idx, imotor]).T
        dfx.columns = ['mark', 'motor']
        dfx = dfx.sort_values(by='mark',ascending=True)
        
        self.stitch_points = dfx
        
    def _calc_scale_factors(self):
        """
        Internal function that calcualtes the scale factor between ``self.variable_motors`` positions.

        """
        dfx = self.stitch_points # Use legacy code
        refl = self.normalized_data # Use legacy code
        scale = 1
        scale_err = 0
        idq = []

        if dfx.empty == False:
            idq.append([dfx.motor.iloc[0], refl[dfx.motor.iloc[0]].iloc[0], scale, 0, 0])

            for j, x in enumerate(dfx['mark']):
                #Qx = refl['Q'].iloc[x] #The Qposition right BEFORE we change motor
                Qstitch = refl['Q'].iloc[x+1] # The Qposition where we average over 'numavg' points to create ratio
                MOTOR_CHANGE = dfx.motor.iloc[j] #The motor that we are tracking for this 'mark' point
                MOTORx = refl[MOTOR_CHANGE].iloc[x] #The motor position that we are currently at
                dummyRlist = [] ##Dummy values to average together
                dummyErrlist = [] ##Dummy errors to average together
                for ii, val in enumerate(refl['Q'].iloc[x+1:]):
                    i = x+ii #Realign index from enumerate to coordinate with the list
                    if val == Qstitch:
                        dummyRlist.append(refl['R'].iloc[x+1+ii])
                        dummyErrlist.append(refl['R_err'].iloc[x+1+ii])
                    else: #Collected all the pnts to be averaged
                        avgR = np.mean(np.array(dummyRlist)) #Find the average of the measurements
                        avgErr = np.sqrt(np.sum(np.square(np.array(dummyErrlist)))/len(dummyErrlist)) #Overly complicated method to add variances of the measurement
                        for iii, stitchpnt in enumerate(refl['Q'].iloc[x-7:x]): ##Look back to find the point we want to average ***THE 5 IS A MAGIC NUMBER!!! BEWARE ALL THOSE WHO USE THIS
                            y=x-7+iii
                            if stitchpnt == Qstitch:
                                MOTORy = refl[MOTOR_CHANGE].iloc[x+1]
                                scalei = avgR/(refl['R'].iloc[y])
                                scale_erri = scalei * ((refl['R_err'].iloc[y] / refl['R'].iloc[y])**2 + (avgErr/avgR)**2 )**(0.5)
                        break    
                        
                    #if val == Qx and refl[MOTOR_CHANGE].iloc[i] != MOTORx:
                        #MOTORy = refl[MOTOR_CHANGE].iloc[i]
                        #scalei = (refl['R'].iloc[i]/refl['R'].iloc[x])
                        #scale_erri= scalei*((refl['R_err'].iloc[i]/refl['R'].iloc[i])**2 + (refl['R_err'].iloc[x]/refl['R'].iloc[x])**2)**(0.5)        
                scale = scale*scalei
                scale_err = scale*((scale_err/scale)**2+(scale_erri/scalei)**2)**(0.5)
                idq.append([MOTOR_CHANGE, MOTORy, scale, scale_err, x+1])
            dfx2 = pd.DataFrame(idq, columns = ['motor', 'value', 'Scale','Scale_Err', 'mark'])
            #dfx.sort_values(by='mark')

            data_scale = dfx2.sort_values(by='mark',ascending=True)
        else:
            data_scale=pd.DataFrame(columns = ['motor', 'value', 'Scale','Scale_Err', 'mark'])
            
        self.scale_factors = data_scale
        
        
    def _stitch_refl(self):
        """
        Internal function that stitches the full profile together.
        
        Returns
        -------
        refl_final : pandas.Dataframe
            Normalized and stitched reflectivity profile. 
        """
        refl = self.normalized_data
        Refl_corr = []
        #Refl_ReMag = []
        numpnts = len(self.scale_factors.mark)
        for i in range(numpnts):
            scale = self.scale_factors.Scale.iloc[i]
            scale_err = self.scale_factors.Scale_Err.iloc[i]
            low_trip = self.scale_factors.mark.iloc[i]
            
            if i==(numpnts-1):
                refl_append = refl.iloc[low_trip:]
            else:
                high_trip = self.scale_factors.mark.iloc[i+1]
                refl_append = refl.iloc[low_trip:high_trip]
            
            for j in range(len(refl_append)):
                Q_corr = refl_append['Q'].iloc[j]
                R_corr = refl_append['R'].iloc[j]/scale#(scale_options.loc[scale_options.value==corr, 'Scale'])
                R_err = R_corr*((refl_append['R_err'].iloc[j]/refl_append['R'].iloc[j])**2 + (scale_err/scale)**2 )**0.5#((scale_options.loc[scale_options.value==corr, 'Scale_Err']/scale_options.loc[scale_options.value==corr, 'Scale'])**2)**0.5)
                   
                Refl_corr.append([Q_corr,R_corr,R_err]) #Removed HOS from the processed data
                
        refl_final = pd.DataFrame(Refl_corr, columns = ['Q','R','R_err'])
    
        return refl_final
                        
    def _update_stats(self, frame=10):
        """
        Quickly update image stats and offsets based on first data-point.
        Common practice has this at frame 10.
        
        """
        
        meta = self.meta.iloc[frame]
        image = self.images[frame]
        
        # Update size of frame
        self.imagex = image.shape[0]
        self.imagey = image.shape[1]
        if self.mask is not None:
            if self.mask.shape != meta['image'].shape:
                print('Error: Mask shape mismatch')
                print('Removing mask and continuing')
                self.mask = np.full(image.shape, True)
            
        # Check if the sample is on the bottom or not.
        if -200 <= meta['Sample Theta'] < 0:
            self.samp_loc = 180
        else:
            self.samp_loc = 0
        # Updated angle_offset.
        # Correct angle should be half 'ccd_theta' (corrected for holder position).
        self.angle_offset = round(meta['CCD Theta']/2 + self.samp_loc - meta['Sample Theta'], 3)
        
        return meta
    
def slice_spot(image, h=20, w=20,
           mask=None, edge_trim=(5, 5),
           diz_threshold=10, diz_size=3
          ): 
    """
    
    Slice an image around the pixel with the highest counts.
    
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
        diz_threshold / diz_size : int
            Dizinger properties to remove 'hot' pixels.
    """
    dx=edge_trim[0]
    dy=edge_trim[1]
    
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
        
        
def slice_dark(image, h=20, w=20,
           mask=None, edge_trim=(5, 5),
           beamspot=(75, 75), darkside='LHS'
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
        mask : np.ndarray, Boolean
            Array of pixels to ignore during calculation. Only pixels set to `True` will be considered.
        darkside ('LHS' or 'RHS')
            Side of the image to take the dark frame.

    """  
    dx=edge_trim[0]
    dy=edge_trim[1]
    #Check if mask is an appropriate input, if not, generate dummy mask and ignore.
    if mask is None:
        mask = np.full(image.shape, True) #No Mask
    elif mask.shape != image.shape:
        mask = np.full(image.shape, True)

    y_spot = beamspot[1]
    if darkside == 'LHS':
        x_dark =  dx  ### box on left side of image
    elif darkside == 'RHS':
        x_dark =  image.shape[0]-dx-w  ### box on right side of image
    else: #Default to RHS
        x_dark =  image.shape[0]-dx-w  ### box on right side of image
    
    y_dark =  y_spot-(h//2)#+dy Already include trim in image_zinged
        
    sl1 = slice(y_dark,y_dark+h)
    sl2 = slice(x_dark,x_dark+w)
    sl_dark = (sl1,sl2)
    image_dark = image[sl_dark]
    
    rect_dark = plt.Rectangle((x_dark,y_dark),w,h,edgecolor='red',facecolor='None')
    return image_dark, rect_dark
        
def load_prsoxr_fits(files):
        """
        Parses every .fits file given in ``files`` and returns the meta and image data
        
        Returns
        -------
        images : list
            List of each image file associated with the .fits
        meta : pd.Dataframe
            pandas dataframe composed of all meta data for each image
        
        """
        temp_meta = {}
        out_images = []
        for i, file in enumerate(files):
            with fits.open(file) as hdul:
                header = hdul[0].header
                del header['COMMENT'] #Doubles length of df for some reason.
                for item in header:
                    temp_meta[item]=header[item]
                out_images.append(hdul[2].data)
            if i == 0:
                out_meta = pd.DataFrame(temp_meta, index=[i])
            else:
                out_meta = out_meta.append(pd.DataFrame(temp_meta, index=[i]))
        
        return out_images, out_meta


