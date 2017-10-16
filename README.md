# JW_lib

RossiterMcLaughlinEffect is a class to calculate line profiles (LPs) during an eclipse of a star by a planet (dark) or a companion star (self luminous)

To get LPs as a function of time:

import JW_lib

lsf = JW_lib.readSpec("tmp.dat.atm") # read in line spread function (LSF) of the instrument 

lsf.wavelength = lsf.converVelocitytoWavelength(lsf.wavelength, lambda_0=2.0) # convert wavelength from velocity to wavelength in micron

JW_lib.RossiterMcLaughlinEffect(lsf=lsf).calcLPSeries(t_arr=None) # t_arr is an array of time between the start and the ending of the eclipse in hours, if None, LPs are calculated at every 0.2 hour time separation. 

The last line will return a series of Spectrum objects (x axis is velocity) that are LPs at different time stamps during an eclipse
