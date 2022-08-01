#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math as math
import numpy as np
import pandas
import scipy.interpolate as si
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


def getmass(mto, imf1, imf2, imfup):

    # Compute mass in stars and remnants (normalized to 1 Msun at t=0)
    # Assume an IMF that runs from 0.08 to 100 Msun.

    # Inputs
    # mlo - lower mass cut-off for integration
    # logage - input log age
    # zh - input metallicity
    # imf1 - IMF_x1 slope
    # imf2 - IMF_x2 slope
    # imfup - high mass slope (Salpeter)

    # Default parameter settings
    bhlim =  40.0  # Mass limit above which star becomes BH
    nslim =   8.5  # Mass above which star becomes NS
    m2    =   0.5  # Break mass for first IMF segment
    m3    =   1.0  # Break mass for second IMF segment
    mlo   =   0.08 # Low-mass cut-off assumed
    imfhi = 100.0  # Upper mass for integration
    
    #---------------------------------------------------------------!
    #---------------------------------------------------------------!

    if mlo > m2:
        print('GETMASS ERROR: mlo>m2')
        return

    getmass  = 0.0

    # normalize the weights so that 1 Msun formed at t=0
    # This comes from defining the three-part piecewise linear IMF, N(m)=-X log(m) + c, 
    # establishing the constant needed for continuity, and integrating m.N(m)dm within the three sections.
    imfnorm = (m2**(-imf1+2)-mlo**(-imf1+2))/(-imf1+2) +                m2**(-imf1+imf2)*(m3**(-imf2+2)-m2**(-imf2+2))/(-imf2+2) +                m2**(-imf1+imf2)*(imfhi**(-imfup+2)-m3**(-imfup+2))/(-imfup+2)

    # stars still alive
    # First the low-mass segment, which is older than the Universe
    getmass = (m2**(-imf1+2)-mlo**(-imf1+2))/(-imf1+2)
    # Now the age-dependent part. mto is the mass of the main-sequence turn off, and is age dependent.

    # if mto < m3, include whole of m2<m<m3
    if mto < m3:
        getmass = getmass + m2**(-imf1+imf2)*(mto**(-imf2+2)-m2**(-imf2+2))/(-imf2+2)

    # otherwise, add the two sections up to mto
    else:
        getmass = getmass + m2**(-imf1+imf2)*(m3**(-imf2+2)-m2**(-imf2+2))/(-imf2+2) +                   m2**(-imf1+imf2)*(mto**(-imfup+2)-m3**(-imfup+2))/(-imfup+2)
     
    # Normalise
    getmass = getmass/imfnorm

    # BH remnants
    # bhlim<M<imf_up leave behind a 0.5*M BH. bhlim=40, set above
    # According to the age-msto relation, a 40Msun star lives < 100,000yr
    getmass = getmass +               0.5*m2**(-imf1+imf2)*(imfhi**(-imfup+2)-bhlim**(-imfup+2))/(-imfup+2)/imfnorm

    # NS remnants
    # nslim<M<bhlim leave behind 1.4 Msun NS
    #  nslim = 8.5 defined above
    # According to the age-msto relation, an 8.5Msun star lives < 10Myr
    getmass = getmass +               1.4*m2**(-imf1+imf2)*(bhlim**(-imfup+1)-nslim**(-imfup+1))/(-imfup+1)/imfnorm

    # WD remnants
    # M<8.5 leave behind 0.077*M+0.48 WD
    # There are two parts that must be added: the 0.077* part, which is a fraction of the MASS integral, and the 'fixed' WD mass, which is a mass contributino based on the NUMBER of stars, so uses the NUMBER integral.

    # If mto lt m3, then must consider WD stars in two segments, up to nslim. Otherwise, only the upper segment.
    if mto < m3:
        getmass = getmass +              0.48*m2**(-imf1+imf2)*(nslim**(-imfup+1)-m3**(-imfup+1))/(-imfup+1)/imfnorm
        getmass = getmass +              0.48*m2**(-imf1+imf2)*(m3**(-imf2+1)-mto**(-imf2+1))/(-imf2+1)/imfnorm
        getmass = getmass +              0.077*m2**(-imf1+imf2)*(nslim**(-imfup+2)-m3**(-imfup+2))/(-imfup+2)/imfnorm
        getmass = getmass +              0.077*m2**(-imf1+imf2)*(m3**(-imf2+2)-mto**(-imf2+2))/(-imf2+2)/imfnorm
    else:
        getmass = getmass +              0.48*m2**(-imf1+imf2)*(nslim**(-imfup+1)-mto**(-imfup+1))/(-imfup+1)/imfnorm
        getmass = getmass +              0.077*m2**(-imf1+imf2)*(nslim**(-imfup+2)-mto**(-imfup+2))/(-imfup+2)/imfnorm

    return getmass


# In[2]:


def getm2l(logage, zh, imf1, imf2, imfup):

    # Compute mass-to-light ratio in several filters (AB mags)

    # INPUTS
    # logage = log10(age)
    # zh = Z/H (0.0=solar)
    # imf1 - IMF_x1 slope
    # imf2 - IMF_x2 slope
    # imfup - high mass slope (should be Salpeter = 2.3)
    
    # Variables
    lsun   = 3.839e33 # Solar luminosity in erg/s
    clight = 2.9979e10 # Speed of light (cm/s)
    mypi   = 3.14159265 # pi
    pc2cm  = 3.08568e18 # cm in a pc
    magsun = np.array([4.64,4.52,4.56,5.14]) # mag of sun in r,I,J,K filters (AB mag)
    
    #---------------------------------------------------------------#
    #---------------------------------------------------------------#

    ##### THE MODEL SHOULD BE GIVEN AS AN INPUT, BUT READ IT HERE FOR TESTING #####
    
    # Read in the model. 
    dir = '/Users/rmcdermid/MQ/Students/Christina/'
    dataset = pandas.read_table(dir+'salp_model_spectrum.txt', sep=' ', header=None, skip_blank_lines=True,comment=';')
    lams = dataset[0].values
    spec = dataset[1].values.astype('float')

    #---------------------------------------------------------------#

    
    # First compute the Main-Sequence Turn Off mass (mto) via relation between mto and (age, metallicity)
    # This was extracted from getm2l.f90, with coefficients from alf_vars.f90
    msto_t0 = 0.33250847
    msto_t1 = -0.29560944
    msto_z0 = 0.95402521
    msto_z1 = 0.21944863
    msto_z2 = 0.070565820
    mto = 10**(msto_t0 + msto_t1 * logage) *           ( msto_z0 + msto_z1 * zh + msto_z2 * zh**2 )

    # Original code gives some different forms of the IMF here, but simplified
    # this to just understand the PyStaff version. imf1/2 come from the fit X_1, X_2
    mass = getmass(mto, imf1, imf2, imfup)
    print('Mass=',mass,'Msun')

    # Read in the filter file from ALF. A few curves are included, but I just take the r-band one
    dataset = pandas.read_table(dir+'filters.dat', delim_whitespace=True, header=None, skip_blank_lines=True,comment=';')
    lamf = dataset[0].values
    filters = dataset[1].values.astype('float')
    
    # Convert the log model to linear lambda via simple interpolation
    interp2=si.interp1d(np.exp(lams), spec, fill_value='extrapolate')
    lin_base_template=interp2(lamf)
    
    # Convert to the 'proper units'. Did not fully understand this, and could not find
    # clear documentation, but the gist is that the model is in frequency units (hence the lambda^2/c)
    # and should be put into some kind of apparent magnitude form (hence the 4piR^2 term), but not sure.
    # Either way, it works....
    aspec  = lin_base_template*lsun/1e6*lamf**2/clight/1.e8/4./mypi/pc2cm**2

    # Integrate the spectrum under the filter curve using trapezoid method.
    mag = np.trapz(aspec*filters/lamf,x=lamf)

    # This should bne the AB mag prediction, comparable to other models
    print('mag(r, AB):',-2.5*np.log10(mag)-48.60)
    
    # Now generate the M/L in solar units.
    if mag <= 0.0:
        m2l = 0.0
    else: 
        mag = -2.5*np.log10(mag)-48.60
        m2l = mass/10**(2./5*(magsun[0]-mag))

    if (m2l > 100.):
        m2l = 0.
                        
    return m2l


# In[17]:


ml = getm2l(1.0, 0.0, 1.0, 1.3, 2.3)
print('M/L(r)=',ml)


# In[ ]:




