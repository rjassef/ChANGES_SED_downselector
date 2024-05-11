import numpy as np
from astropy.table import Table

#This code implements the stellar locus cuts described in the "remove_stars.ipynb" notebook.

def read_davenport():

    #Read the stellar locus fits form Davenport et al. (2014). We will use the following colors for the analysis: 
    #g-i, r-i, r-z, z-W1, r-W1
    dv14 = Table.read("Davenport14/table1.txt",format='ascii')
    dv14_nir = Table.read("Davenport14/table2.txt",format='ascii')
    sl_tab = Table()
    sl_tab['g-i']  = dv14['g-i']
    sl_tab['r-i']  = dv14['r-i']
    sl_tab['r-z']  = dv14['r-i'] + dv14['i-z']
    sl_tab['z-w1'] = dv14['z-J'] + dv14['J-H'] + dv14['H-Ks'] + dv14_nir['Ks-W1']
    sl_tab['r-w1'] = dv14['r-i'] + dv14['i-z'] + dv14['z-J'] + dv14['J-H'] + dv14['H-Ks'] + dv14_nir['Ks-W1']

    return sl_tab

def remove_stars(tab, no_r_survey=False):

    #Start by reading the stellar locus. 
    sl_tab = read_davenport()

    #####
    # Proper Motions
    #####

    #Now, set the proper motion cut. This is that in order to be selected, a source must have a proper motion detected below SNR=5 (and possible masked as a non-detection).
    pm_err = ((tab['pmdec_error'] * tab['pmdec']/tab['pm'])**2 + (tab['pmra_error'] * tab['pmra']/tab['pm'])**2)**0.5
    pm_snr = tab['pm']/pm_err
    pm_cond = (pm_snr<5) | (pm_snr.mask)

    #Now, determine the colors cuts. 
    
    #####
    # No r-band survey - Needed for this survey, but not justified for the rest. 
    #####
    if no_r_survey:
        
        #Interpolate the stellar locus tab to create a limit in z-W1 for the g-i color of each source.
        zw1_lim = np.interp(tab['mag_auto_g']-tab['mag_auto_i'], sl_tab['g-i'], sl_tab['z-w1']) + 0.5
        sl_nr_cut = tab['mag_auto_z']-tab['w1mpro']<zw1_lim

        #Additionally, since the cut only makes sense for sources detected in all four bands,
        #we should require a detection on them. 
        for band in ['mag_auto_g', 'mag_auto_i', 'mag_auto_z', 'w1mpro']:
            sl_nr_cut = sl_nr_cut & (tab[band]<99.)        

    #####
    # First color cut
    #####
    #Interpolate the stellar locus tab to create a limit in z-W1 for the r-i color of each source.
    zw1_lim = np.interp(tab['mag_auto_r']-tab['mag_auto_i'], sl_tab['r-i'], sl_tab['z-w1']) + 0.5
    sl_cut1 = tab['mag_auto_z']-tab['w1mpro']<zw1_lim

    #Additionally, since the cut only makes sense for sources detected in all four bands,
    #we should require a detection on them. 
    for band in ['mag_auto_r', 'mag_auto_i', 'mag_auto_z', 'w1mpro']:
        sl_cut1 = sl_cut1 & (tab[band]<99.)

    ####
    # Second color cut
    ####
    #Interpolate the stellar locus tab to create a limit in z-W1 for the r-i color of each source.
    zw1_lim = np.interp(tab['mag_auto_r']-tab['mag_auto_i'], sl_tab['r-i'], sl_tab['z-w1']) + 1.0
    sl_cut2 = (tab['mag_auto_z']-tab['w1mpro']<zw1_lim) & (tab['mag_auto_r']-tab['mag_auto_i']>1.0)

    #Additionally, since the cut only makes sense for sources detected in all four bands,
    #we should require a detection on them. 
    for band in ['mag_auto_r', 'mag_auto_i', 'mag_auto_z', 'w1mpro']:
        sl_cut2 = sl_cut2 & (tab[band]<99.)

    ####
    # Third color cut
    ####
    #Interpolate the stellar locus tab to create a limit in z-W1 for the r-i color of each source.
    rw1_lim = np.interp(tab['mag_auto_r']-tab['mag_auto_i'], sl_tab['r-i'], sl_tab['r-w1']) + 0.5
    sl_cut3 = (tab['mag_auto_r']-tab['w1mpro']<rw1_lim) & (tab['mag_auto_r']-tab['mag_auto_i']>1.0)

    #Additionally, since the cut only makes sense for sources detected in all three bands,
    #we should require a detection on them. 
    for band in ['mag_auto_r', 'mag_auto_i', 'w1mpro']:
        sl_cut3 = sl_cut3 & (tab[band]<99.)

    #Then the full requirement is
    sel_cond = pm_cond & (~sl_cut1) & (~sl_cut2) & (~sl_cut3)
    if no_r_survey:
        sel_cond = sel_cond & (~sl_nr_cut)

    return tab[sel_cond]


