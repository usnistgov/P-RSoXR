Preparing to collect data
##########################

The next step is to align each sample and generate a script that will be used to collect data. It is recommended to use the provided `prsoxr_script_gen.py` to assist in generating a runfile. The following steps will outline how to use the script. It only requires `numpy` and `pandas` to be installed.

Initial sample information
***************************
The first steps to generating a runfile is to fill out the "Setup Parameters" section. This will require some basic bookeeping that will later allow you to sort your new data.

Begin with information about the samples mounted on the plate::
	
	"""
	Setup Parameters
	"""
	save_dir = '.../P-RSoXR/' # Location to save script
	save_name = 'test_spol' # Scipt name

	#Store sample names and energies you want to run
	sample_list = {} #Do not change this line
	sample_list['Samp_A'] = [270.0, 284.5, 287.3, 288.7]
	sample_list['Samp_B'] = [270.0, 284.3, 288.7]
	sample_list['Samp_C'] = [250.0, 283.5, 284.8, 285.3]
	
	#External energy calibration
	EnergyOffset = 0 # [eV]
	
You will need to set a directory (**save_dir**) that you want to save the script and a descriptive name (**save_name**). Basic sample information is stored in the ``sample_list`` dictionary. Make a new key for each sample and a corresponding list of each energy that you would like to run on that sample.

The energies that you pick will want to be the true energies that you want to run. If you wish to apply some external offset, you can set ``EnergyOffset`` to something other than 0. The final script will then run::
	
	energy_run = energy_in_list + EnergyOffset

.. note::
	This script will at no point ask for an input polarization because we recommend running each polarization independently. At beamline 11.0.1.2, changing the polarization state makes the beam intensity semi-unstable for at least 30 minutes. Before any alignment, set the ``EPU Polarization`` motor to your desired state.

	Once this script has finished running, change the polarization, wait 30 minutes, then run it again. 
		
Once each sample has been defined, motor positions need to be setup. Each script input will consist of a list with each element corresponding to a sample. Input them in the order you wish to collect data. We will define the inputs here and describe how you collect them in the following sections::

	# Inputs to run sample plate
	SampleOrder = ['Samp_A', 'Samp_B', 'Samp_C'] # Order that you want to run samples.
	# Motor positions
	XPosition = [36.52, 23.36, 0] # 'Sample X' motor position
	XOffset = [0.15, 0.15, 0.15] # Offset to translate beam on sample after each energy.
	YPosition = [-3.2150, -1.9750, 0] # 'Sample Y' motor position
	ZPosition = [-0.27, -0.36, 0] # 'Sample Z' motor position
	ZFlipPosition = [0.21, 0.15, 0] # 'Sample Z' motor position when 'Sample Theta' is flipped 180deg
	ReverseHolder = [0, 0, 0] #Set to True (1) for a sample that is on the reverse side of the holder
	ZDirectBeam = [-2, -2, -1] #'Sample Z' motor positions to access the direct beam
	ThetaOffset = [0, -0.55, 0] #'Sample Theta' offset if measuring multiple samples

	##
	SampleThickness = [250, 250, 250] #Approximate film thickness [A], used to determine dq
	LowAngleDensity = [15, 15, 15] #Approximate number of points per fringe at low angles
	HighAngleDensity = [9, 9, 9] #Approximate number of points per fringe at high angles
	AngleCrossover = [10, 10, 10] #Angle [deg] to cross from low -> high density
	##
	OverlapPoints = [3, 3, 3] #Number of overlap points upon changing motor. Used for stitching
	CheckUncertainty = [3, 3, 3] #Number of points for assessing error at each change in motor conditions. Used for error reduction
	HOSBuffer = [2, 2, 2] #Number of points to add upon changing HOS to account for  motor movement. 
	I0Points = [10, 10, 10] #Number of I0 measureents taken at the start of the scan. Used to calculate direct beam uncertainty
	
* Motor Positions:
	#. **SampleOrder**: This is a list of sample names. They are in the order that you want to run. Use the same key phrases used in defining your sample dictionary.
	#. **XPosition**: This is the ``Sample X`` motor position at the spot of the sample to be measured.
	#. **XOffset**: This is an offset that will laterally translate your ``Sample X`` position at each energy. Typically set slightly larger than the size of th beam to measure a fresh spot at each energy. Set to 0 if do not want to move the beam while measuring (not recommended)
	#. **YPosition**: This is the ``Sample Y`` motor position at the spot of the sample to be measured.
	#. **ZPosition**: This is the ``Sample Z`` motor position that cuts the beam in half when ``Sample Theta = 0``
	#. **ZFlipPosition**: This is the ``Sample Z`` motor position that cuts the beam in half when ``Sample Theta = 180``
	#. **ReverseHolder**: This is a binary option that is set on whether the sample is on the top (0) or bottom (1) of the plate.
	#. **ZDirectBeam**: This is a ``Sample Z`` position that moves the entire plate to enable measuring the direct beam.
	#. **ThetaOffset**: This is an angle offset to be measured for each sample.

The next settings define the q-spacing to be measured. This is defined by an approximate number of measurements that we want to collect per fringe. If we assume that the highest frequency fringe spacing will depend on the total film thickness, ``L = 2*np.pi/thick``, the q-spacing is given by ``dq = L/N`` where N is the chosen point density.

* Settings for point density:
	#. **SampleThickness**: An approximate sample film thickness in [A]. This will be referenced to determine dq
	#. **LowAngleDensity**: Number of points per fringe at low angles. Typically high to resolve critical angle
	#. **HighAngleDensity**: Number of points per fringe at high angles. Typically lower once past the critical angle.
	#. **AngleCrossOver**: Angle [deg] to cross from low to high density.
	
The final settings is to help with measurement statistics and buffer the measurement to allow for slower motors to reach their positions.

* Settings for Statistics:
	#. **OverlapPoints**: Number of points to repeat when upstream optics are moved to increase flux. This will be used to stitch the data together.
	#. **CheckUncertainty**: Number of points to repeat to assess error when changing ``Higher Order Suppressor`` or ``Horizontal Exit Slit Size``.
	#. **HOSBuffer**: Number of points to add, and later discard, to make sure the 'Higher Order Suppressor`` has reached its new position.
	#. **I0Points**: Number of times to measure the direct beam measurements at the start of each energy. This value will be averaged and the standard error will then be propogated into measurement uncertainty.
	
Sample Alignment
****************
The following steps will outline the process to align the axis of rotation of each sample.

.. note::
	Samples mounted on the underside of the sample plate will require a slightly different alignment procedure. Subtract **180** from all ``Sample Theta`` positions. If this subtraction causes the angle to be **-360**, set ``Sample Theta = 0``
 
* Locate the sample:
	#.	Set ``Sample Theta = 90``
	#.	Move ``Sample X`` and ``Sample Y`` until you reach the position on the sample you wish to measure.
	
		* Record these motor positions in the script: **XPosition** and **YPosition** 
		
	#.	Set ``Sample Theta = 0`` and find a ``Sample Z`` position that allows the direct beam to pass unobstructed. (Typically around (-5 mm) when ``Sample Theta = 0`` or (5 mm) when ``Sample Theta = -180``)
	
		* Record the ``Sample Z`` position in the script: **ZDirectBeam**
		* Remember to set **ZFlipPosition = 1** if the sample is on the bottom of the plate.
		
	#.	Set ``CCD Theta = 0`` and take an image of the direct beam with an exposure time that does not saturate (start with 0.001 [s])
	
		* Record the total number of counts
		* Measure the dark counts with the same exposure
		* Calculate the direct beam counts: I0 = light - dark
	
	#.	Set the crosshairs on the CCD display to the center of the beamspot (if not already done)
		
* Calibrate the axis of rotation:

This point in the procedure is an iterative process. It may take 3 or 4 iterations until the sample is properly aligned.
		
	#.	With ``CCD Theta = 0`` and ``Sample Theta = 0`` run a ``Sample Z`` single-motor scan that raises the plate into the beam. We are looking for the motor position that cuts the flux in half.
		
		* Monitor the total counts as the motor moves. Use the previously calculated I0 as a reference for the full flux.
		
	#.	Set ``Sample Z`` to the position that cuts the beam in half, ``Sample theta = 4`` and ``CCD Theta = 8`` and snap an image. 
	
		.. note::
			If the beam is properly cut in half, the CCD image will only have signal **above** the crosshairs.
	
	    *	If the beamspot is **not** within the crosshairs:
		
				* Run a ``Sample Theta`` single-motor scan until the beamspot aligns with the crosshairs. Manually adjust the final position if necessary.
				
				* If this is the **first** sample:
				
					* Set the new ``Sample Theta == 4``
					
				* Otherwise:
				
					* Make note of the angle offset. Add this value to ``Sample Theta`` during the remaining calibration (it can be negative!).
					
			* Return to step 1
			
	    *	If the beamspot is within the crosshairs:
		
				* Set ``Sample Theta = 10`` and ``CCD Theta = 20`` and snap an image.
				
				*	If the beamspot is *not* within the crosshairs
				
					* Adjust ``Sample Theta`` until it becomes aligned
					* If this is the **first** sample:
					
						* Set the new ``Sample Theta == 10``
						
					* Otherwise:
					
						* Make note of the angle offset. Add (or subtract) this value to ``Sample Theta`` during the remaining calibration (it can be negative!).
						
					* Return to step 1
					
				*	If the beamspot does not move (or is acceptable)
				
					* Record ``Sample Z`` in the script: **ZPosition**
					* Record your angle offset in the script: **ThetaOffset**
					
	*	Having aligned ``Sample Theta``, set ``Sample Theta = -180`` and ``CCD Theta = 0``.
	
	*	Run a ``Sample Z`` single-motor scan that raises the plate into the beam. Find the position that cuts it in half and record it in the script as **ZFlipPosition**
	
		.. note::
			This motor position will typically be the opposite sign of **ZPosition** and the approximate difference between the two values will be the substrate thickness.
			
Repeat this process for each sample.

Adjust incident flux 
**********************
Having aligned each sample, the next step is to setup upstream optics and exposure times to progressively increase flux while the angle is increased. Beamline 11.0.1.2 can achieve up to 7 orders of magnitude by adjusting the following:

#.	``Higher Order Suppressor`` : This is a 4-bounce mirror that is designed to eliminate higher order light. Keep it above *7.5 [deg]*
#.	``Horizontal Exit Slit Size`` : This is an upstream beam-shaping slit. At non-resonant energies, this is typically cut down to *150 [mm]*. For more flux, it can be increased to *1500 [mm]*. This is usually a binary operation.
#.	``Exposure`` : How long do you want to dwell at a single position. It is good practice to never go above 1 [s] exposures. The amount of flux gained beyond that is minimal compared to the increase in beam damage and the overall length of the experiment.

We will manually survey the full theta scan that we want to run for each sample (and energy). Once the flux has dropped by some threshold, we will adjust one of the above motors. This is typically done in the order: ``Higher Order Suppressor``, ``Exposure``, and finally ``Horizontal Exit Slit Size``.

Within *prsoxr_script_gen.py*, you will want to copy the following block of code for every sample::

	"""
	COPY THIS BLOCK OF CODE FOR EACH SAMPLE
	"""
	#########################################
	#####            Sample 0           #####
	#########################################
	DEG = [] # Angles to change settings
	HOS = [] # HOS positions
	HES = [] # HES positions
	EXP = [] # Exposures
	#########################################

	#########################################
	#####           Energy 1            #####
	#########################################
	DEG.append([1, 4, 10, 15, 20, 25, 30, 40, 75])
	HOS.append([12, 11, 10, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5])
	HES.append([150, 150, 150, 150, 150, 150, 150, 150, 150])
	EXP.append([0.001, 0.001, 0.001, 0.001, 0.1, 0.5, 0.5, 0.5, 0.5])

	#########################################
	#####           Energy 2            #####
	#########################################
	DEG.append([1, 4, 6, 10, 12, 15, 20, 25, 30, 40, 60])
	HOS.append([12, 11.5,  12, 10, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5])
	HES.append([1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500])
	EXP.append([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1])

	#########################################
	#####           Energy 3            #####
	#########################################
	DEG.append([1, 4, 6, 10, 12, 15, 20, 25, 30, 40, 60])
	HOS.append([12, 11.5,  12, 10, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5])
	HES.append([1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500])
	EXP.append([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1])

	#########################################
	#####           Energy 4            #####
	#########################################
	DEG.append([1, 4, 6, 10, 12, 15, 20, 25, 30, 40, 60])
	HOS.append([12, 11.5,  12, 10, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5])
	HES.append([1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500])
	EXP.append([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1])

	#Compile inputs
	motors = zip(DEG, HOS, HES, EXP)
	temp_list = []
	columns = ['Angle', 'HOS', 'HES', 'EXPOSURE']
	for deg, hos, hes, exp in motors:
		temp_list.append(pd.DataFrame({'Angle':deg, 'HOS':hos, 'HES':hes, 'EXPOSURE':exp}))    
	VariableMotors.append(temp_list)
	"""
	STOP COPYING HERE
	"""
	
The example given is for a sample that will be measured at four energies. Under each four energies are lists correspond to the 3 motors discussed earlier and **DEG** which corresponds to ``Sample Theta``. In these lists we will compile a set of angles, that once reached in our 'theta-2theta' scan, will update the upstream optics to increase flux.

This is a process that you will want to complete for each energy (and likely polarization). 

#.	Always begin with the following settings: ``Higher Order Suppressor = 12``, ``Exposure = 0.001``, and ``Horizontal Exit Slit Size = 150`` (or ``Horizontal Exit Slit Size = 1500`` if above 285 eV)
#.	Set ``Sample Theta = 0`` , ``CCD Theta = 0``, ``Sample Z`` to the direct beam location, and ``Beamline Energy`` to the target energy (be sure to include the **EnergyOffset**). Take a snap of the beam.
	
	* For any samples with a corresponding **AngleOffset**, be sure to add it to every ``Sample Theta`` used during this alignment step.
	
	* For samples mounted on the bottom of the plate, be sure to subtract **180** from each ``Sample Theta``

#.	Adjust the ``Exposure`` until you maximize counts (or nearly so) without saturating any pixels.

	* Set this value into the first element of the **EXP** list. In the example, for 'Energy 1', this value is *0.001*. We recommend to only use this exposure until the ``Higher Order Suppressor == 7.5``
	
#.	Set ``Sample Theta = 4`` and ``CCD Theta = 8`` and snap an image with identical settings.

	* The beam should still be visible, and with decent counts. Pixels should have at least hundreds of counts over the background.
	* If the beam is completely gone, or too bright, adjust the ``Sample Theta`` position until the beam is at a good intensity to change settings. Remember to update the ***DEG** list to this new position.
	
#.	Adjust the ``Higher Order Suppressor`` at this new ``Sample Theta`` position until you have maximized counts.

	* Set this value into the second element of **HOS**.

#.	Repeat steps 4 and 5 for ``Sample Theta = [6, 10, 12, 15, 20, 25, 30, 40, 60]``. Always set `CCD Theta`` to double ``Sample Theta`` in order to measure the beam.
	
	* Once ``Higher Order Suppressor == 7.5``, begin increasing ``Exposure``. At ``Exposure == 1`` it is unlikely that any increase will get you better data without going to 10s or higher. It is often not worth collecting this data due to the amount of time it will take.

The provided ``Sample Theta`` positions are simply a suggestion. Change them or add more as you see fit for your samples.

Running the script
*****************************
Once you have filled in all the necessary information, run the python script. Two files should be generated:

#.	A runfile: 'save_name'.txt
#.	A header file: 'save_name'_HEADER.txt

The header file keeps information about the samples and the energies that you want to run. This is for sorting the output data once it has been collected. The runfile needs to be transfered onto the beamline computer to run.

Set the acquisition parameters to ``From File``, select the runfile, click ``Start``











