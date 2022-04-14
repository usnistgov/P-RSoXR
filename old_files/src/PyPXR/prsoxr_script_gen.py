"""
Runfile to generate a motor-position script to be read by ALS beamline 11.0.1.2. 
"""

#Initializations
import numpy as np
import pandas as pd
import math


"""
Setup Parameters
"""
save_dir = '.../P-RSoXR/' # Location to save script
save_name = 'test' # Scipt name

#Store sample names and energies you want to run
sample_list = {} #Do not change this line
sample_list['Samp_A'] = [270.0, 284.5, 287.3, 288.7]
sample_list['Samp_B'] = [270.0, 284.3, 288.7]
sample_list['Samp_C'] = [250.0, 283.5, 284.8, 285.3]

#External energy calibration
EnergyOffset = 0 # [eV]

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

#Initialize compilation lists:
VariableMotors = []
MotorPositions = {}

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
DEG.append([1, 6, 10, 15, 20, 25, 30, 40, 75])
HOS.append([12, 11, 10, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5])
HES.append([150, 150, 150, 150, 150, 150, 150, 150, 150])
EXP.append([0.001, 0.001, 0.001, 0.001, 0.1, 0.5, 0.5, 0.5, 0.5])

#########################################
#####           Energy 2            #####
#########################################
DEG.append([1, 4, 6, 10, 12, 15, 20, 25, 30, 40, 75])
HOS.append([12, 11.5,  12, 10, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5])
HES.append([1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500])
EXP.append([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1])

#########################################
#####           Energy 3            #####
#########################################
DEG.append([1, 4, 6, 10, 12, 15, 20, 25, 30, 40, 75])
HOS.append([12, 11.5,  12, 10, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5])
HES.append([1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500])
EXP.append([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1])

#########################################
#####           Energy 4            #####
#########################################
DEG.append([1, 4, 6, 10, 12, 15, 20, 25, 30, 40, 75])
HOS.append([12, 11.5,  12, 10, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5])
HES.append([1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500])
EXP.append([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1])

#Consolodate information
motors = zip(DEG, HOS, HES, EXP)
temp_list = []
columns = ['Angle', 'HOS', 'HES', 'EXPOSURE']
for deg, hos, hes, exp in motors:
    temp_list.append(pd.DataFrame({'Angle':deg, 'HOS':hos, 'HES':hes, 'EXPOSURE':exp}))    
VariableMotors.append(temp_list)
"""
STOP COPYING HERE
"""
#Add samples here ~~






"""
DON'T ADD SAMPLES BEYOND THIS POINT
"""
#Setting up meta-data
MotorPositions['XPosition'] = Xposition
MotorPositions['YPosition'] = Xposition
MotorPositions['ZPosition'] = Xposition
MotorPositions['XOffset'] = XOffset
MotorPositions['ZFlipPosition'] = ZFlipPosition
MotorPositions['ZDirectBeam'] = ZDirectBeam
MotorPositions['ThetaOffset'] = ThetaOffset
MotorPositions['ReverseHolder'] = ReverseHolder
MotorPositions['SampleThickness'] = SampleThickness
MotorPositions['LowAngleDensity'] = LowAngleDensity
MotorPositions['HighAngleDensity'] = HighAngleDensity
MotorPositions['AngleCrossover'] = AngleCrossover
MotorPositions['OverlapPoints'] = OverlapPoints
MotorPositions['CheckUncertainty'] = CheckUncertainty
MotorPositions['HOSBuffer'] = HOSBuffer
MotorPositions['I0Points'] = I0Points
df_motor = pd.DataFrame(MotorPositions)

#Function that will generate beamline inputs
def AngleRunGeneration(MotorPositions, VariableMotors, Energy):
    #Constants ## https://www.nist.gov/si-redefinition
    SOL = 299792458 #m/s
    PLANCK_JOULE = 6.6267015e-34 #Joule s
    ELEMCHARGE =  1.602176634e-19 #coulombs
    PLANCK = PLANCK_JOULE / ELEMCHARGE #eV s
    meterToAng = 10**(10)
    ##Initialization of needed components
    Wavelength = SOL * PLANCK * meterToAng / Energy
    AngleNumber = VariableMotors['Angle'].nunique()
    XPosition = MotorPositions['XPosition']
    YPosition = MotorPositions['YPosition']
    ZPosition = MotorPositions['ZPosition']
    Z180Position = MotorPositions['ZFlipPosition']
    Zdelta = ZPosition - Z180Position
    ThetaOffset = MotorPositions['ThetaOffset']
    SampleThickness = MotorPositions['SampleThickness']
    LowAngleDensity = MotorPositions['LowAngleDensity']
    HighAngleDensity = MotorPositions['HighAngleDensity']
    AngleCrossover = MotorPositions['AngleCrossover']
    OverlapPoints = int(MotorPositions['OverlapPoints'])
    for i in range(AngleNumber-1):
        if i ==0: # starts the list
            ##Calculate the start and stop location for Q
            AngleStart = VariableMotors['Angle'].iloc[i] # All of the relevant values are in terms of angles, but Q is calculated as a check
            AngleStop = VariableMotors['Angle'].iloc[i+1]
            QStart = 4*math.pi*math.sin(AngleStart*math.pi/180)/Wavelength
            QStop = 4*math.pi*math.sin(AngleStop*math.pi/180)/Wavelength
            
            if AngleStop <= AngleCrossover:
                AngleDensity=LowAngleDensity
            else:
                AngleDensity=HighAngleDensity
                
            #Setup dq in terms of an approximate fringe size (L = 2*PI/Thickness)
            #Break it up based on the desired point density per fringe
            dq=2*math.pi/(SampleThickness*AngleDensity)
            QPoints = math.ceil((QStop-QStart)/dq) #Number of points to run is going to depend on fringe size
            QList = np.linspace(QStart,QStop,QPoints).tolist() #Initialize the QList based on initial configuration
            SampleTheta = np.linspace(AngleStart,AngleStop,QPoints) ##Begin generating list of 'Sample Theta' locations to take data
            CCDTheta = SampleTheta*2 #Make corresponding CCDTheta positions
            SampleTheta = SampleTheta+ThetaOffset #If running multiple samples in a row this will offset the sample theta based on alignment ##CCD THETA SHOULD NOT BE CHANGED
            
            #Check what side the sample is on. If on the bottom, sample theta starts @ -180
            #if MotorPositions['ReverseHolder'] == 1:
            #    SampleTheta=SampleTheta-180 # for samples on the backside of the holder, need to check and see if this is correct
            
            SampleX=[XPosition]*len(QList)
            BeamLineEnergy=[Energy]*len(QList)
            SampleY = YPosition+Zdelta/2+Zdelta/2*np.sin(SampleTheta*math.pi/180) #Adjust 'Sample Y' based on the relative axis of rotation
            SampleZ = ZPosition+Zdelta/2*(np.cos(SampleTheta*math.pi/180)-1) #Adjust 'Sample Z' based on the relative axis of rotation

            #Convert numpy arrays into lists for Pandas generation
            SampleTheta=SampleTheta.tolist()
            SampleY=SampleY.tolist()
            SampleZ=SampleZ.tolist()
            CCDTheta=CCDTheta.tolist()
            
            #Generate HOS / HES / Exposure lists for updating flux
            HOSList=[VariableMotors['HOS'].iloc[i]]*len(QList)
            HESList=[VariableMotors['HES'].iloc[i]]*len(QList)
            ExposureList=[VariableMotors['EXPOSURE'].iloc[i]]*len(QList)
            
            #Adding points to assess the error in beam intensity given new HOS / HES / Exposure conditions
            for d in range(int(MotorPositions['I0Points'])):
                QList.insert(0,0)
                #ThetaInsert = 0 if MotorPositions['ReverseHolder']==0 else -180
                SampleTheta.insert(0,0)
                CCDTheta.insert(0,0)
                SampleX.insert(0,SampleX[d])
                SampleY.insert(0,YPosition)
                SampleZ.insert(0,MotorPositions['ZDirectBeam'])
                HOSList.insert(0,HOSList[d])
                HESList.insert(0, HESList[d])
                ExposureList.insert(0,ExposureList[d])
                BeamLineEnergy.insert(0,BeamLineEnergy[d])
            
        
        else: # for all of the ranges after the first set of samples
            ##Section is identical to the above
            AngleStart=VariableMotors['Angle'].iloc[i]
            AngleStop=VariableMotors['Angle'].iloc[i+1]
            QStart=4*math.pi*math.sin(AngleStart*math.pi/180)/Wavelength
            QStop=4*math.pi*math.sin(AngleStop*math.pi/180)/Wavelength
            
            if AngleStop <= AngleCrossover:
                AngleDensity=LowAngleDensity
            else:
                AngleDensity=HighAngleDensity
                
            dq=2*math.pi/(SampleThickness*AngleDensity)
            QPoints=math.ceil((QStop-QStart)/dq)
            QListAddition=np.linspace(QStart,QStop,QPoints).tolist()
            SampleThetaAddition=np.linspace(AngleStart,AngleStop,QPoints).tolist()
            ##Calculate the points that are used to stitch datasets
            #p+2 selects the appropriate number of points to repeat without doubling at the start of the angle range.
            #Compensate the number of points by reducing OverlapPoints down by 1 (Nominally at 4)
            for p in range(OverlapPoints):
                QListAddition.insert(0,QList[-1*(p+2)]) #Add to Qlist
                SampleThetaAddition.insert(0,SampleTheta[-1*(p+2)]-ThetaOffset) #Add to Sample Theta List ###QUICK CHANGE! REMOVE SAMPLE OFFSET TO ADDITION
            SampleThetaAdditionArray=np.asarray(SampleThetaAddition) #Convert back to numpy array
            
            CCDThetaAddition=SampleThetaAdditionArray*2 #Calculate the CCD theta POsitions
            CCDThetaAddition=CCDThetaAddition.tolist() #Convert to list
            SampleThetaAdditionArray=SampleThetaAdditionArray+ThetaOffset #Account for theta offset
            SampleThetaAddition = SampleThetaAdditionArray.tolist()
            #Check what side the sample is on. If on the bottom, sample theta starts @ -180
            #if MotorPositions['ReverseHolder']==1:
            #    SampleThetaAdditionArray=SampleThetaAdditionArray-180
            
            SampleXAddition=[XPosition]*len(QListAddition)
            BeamLineEnergyAddition=[Energy]*len(QListAddition)
            SampleYAddition=YPosition+Zdelta/2+Zdelta/2*np.sin(SampleThetaAdditionArray*math.pi/180)
            SampleZAddition=ZPosition+Zdelta/2*(np.cos(SampleThetaAdditionArray*math.pi/180)-1) 
            SampleYAddition=SampleYAddition.tolist()
            SampleZAddition=SampleZAddition.tolist()
            
            #Generate HOS / HES / Exposure lists for updating flux
            HOSListAddition=[VariableMotors['HOS'].iloc[i]]*len(QListAddition)
            HESListAddition = [VariableMotors['HES'].iloc[i]]*len(QListAddition)
            ExposureListAddition=[VariableMotors['EXPOSURE'].iloc[i]]*len(QListAddition)
            
            #Check to see if any of the variable motors have moved to add buffer points
            if VariableMotors['HOS'].iloc[i] != VariableMotors['HOS'].iloc[i-1] or VariableMotors['HES'].iloc[i] != VariableMotors['HES'].iloc[i-1] or VariableMotors['EXPOSURE'].iloc[i] != VariableMotors['EXPOSURE'].iloc[i-1]:
            #If a change is made, buffer the change with points to judge new counting statistics error and a few points to buffer the motor movements. 
            #Motor movements buffer is to make sure motors have fully moved before continuing data collection / may require post process changes
            #Adding points to assess the error in beam intensity given new HOS / HES / Exposure conditions
                for d in range(int(MotorPositions['CheckUncertainty'])):
                    QListAddition.insert(0,QListAddition[d])
                    SampleThetaAddition.insert(0,SampleThetaAddition[d])
                    CCDThetaAddition.insert(0,CCDThetaAddition[d])
                    SampleXAddition.insert(0,SampleXAddition[d])
                    SampleYAddition.insert(0,SampleYAddition[d])
                    SampleZAddition.insert(0,SampleZAddition[d])
                    HOSListAddition.insert(0,HOSListAddition[d])
                    HESListAddition.insert(0,HESListAddition[d])
                    ExposureListAddition.insert(0,ExposureListAddition[d])
                    BeamLineEnergyAddition.insert(0,BeamLineEnergyAddition[d])

                #Adding dummy points to beginning of to account for HOS movement 
                for d in range(int(MotorPositions['HOSBuffer'])):
                    QListAddition.insert(0,QListAddition[d])
                    SampleThetaAddition.insert(0,SampleThetaAddition[d])
                    CCDThetaAddition.insert(0,CCDThetaAddition[d])
                    SampleXAddition.insert(0,SampleXAddition[d])
                    SampleYAddition.insert(0,SampleYAddition[d])
                    SampleZAddition.insert(0,SampleZAddition[d])
                    HOSListAddition.insert(0,HOSListAddition[d])
                    HESListAddition.insert(0,HESListAddition[d])
                    ExposureListAddition.insert(0,ExposureListAddition[d])
                    BeamLineEnergyAddition.insert(0,BeamLineEnergyAddition[d])
            
            QList.extend(QListAddition)
            HOSList.extend(HOSListAddition)
            HESList.extend(HESListAddition)
            ExposureList.extend(ExposureListAddition)
            SampleTheta.extend(SampleThetaAddition)
            CCDTheta.extend(CCDThetaAddition)
            SampleX.extend(SampleXAddition)
            SampleY.extend(SampleYAddition)
            SampleZ.extend(SampleZAddition)
            BeamLineEnergy.extend(BeamLineEnergyAddition)
        
        #Check what side the sample is on. If on the bottom, sample theta starts @ -180
    if MotorPositions['ReverseHolder'] == 1:
        SampleTheta=[theta-180 for theta in SampleTheta] # for samples on the backside of the holder, need to check and see if this is correct

            
    return (SampleX, SampleY, SampleZ, SampleTheta, CCDTheta, HOSList, HESList, BeamLineEnergy, ExposureList, QList)

##Generate the compiled dataframe for each sample

AdjustableMotors = ['Sample X', 'Sample Y', 'Sample Z',
                    'Sample Theta', 'CCD Theta', 'Higher Order Suppressor',
                    'Horizontal Exit Slit Size', 'Beamline Energy', 'Exposure'
                    ]
RunFile = pd.DataFrame(columns = AdjustableMotors)
#NumSamples = len(VariableMotors) #The number of variable motor scans corresponding to each sample location
#NumEnergies = len(Energy) #The number of energies for each sample

for i, Samp in enumerate(VariableMotors):
    en_list = sample_list[SampleOrder[i]]
    for j, loc in enumerate(Samp):
        sampdf = pd.DataFrame()
        samp_energy = np.round(float(en_list.iloc[j] + EnergyOffset),2)
        MotorPos_En = AngleRunGeneration(df_motor.iloc[i], loc, samp_energy)
        sampdf['Sample X'] = np.round(np.round(MotorPos_En[0], 4) + XOffset[i]*j , 4)
        sampdf['Sample Y'] = np.round(MotorPos_En[1], 4)
        sampdf['Sample Z'] = np.round(MotorPos_En[2], 4)
        sampdf['Sample Theta'] = np.round(MotorPos_En[3], 4)
        sampdf['CCD Theta'] = np.round(MotorPos_En[4], 4)
        sampdf['Higher Order Suppressor'] = MotorPos_En[5]
        sampdf['Horizontal Exit Slit Size'] = MotorPos_En[6]
        sampdf['Beamline Energy'] = MotorPos_En[7]
        sampdf['Exposure'] = MotorPos_En[8]

        RunFile = RunFile.append(sampdf, ignore_index=True)
        
#Cleanup
runfile_name = save_dir + save_name

HeaderFile = pd.DataFrame(sample_list)
RunFile.to_csv((runfile_name + '.txt'), index=False, sep='\t')
HeaderFile.to_csv((runfile_name+'_HEADER.txt'), index=False, sep='\t')

###Cleanup the output -- Definitly better ways to do this....
with open((runfile_name + '.txt'), 'r') as f: #Loads the file into memory
    lines = f.readlines()
    
lines[0] = lines[0].replace('\tExposure' , '') #Remove the 'Exposure' header
lines[-1] = lines[-1].replace('\n', '') #Remove the last carriage return

with open((runfile_name + '.txt'), "w") as f: #Writes it back in
    f.writelines(lines)
    
del lines #Remove it from memory (it can be large)



