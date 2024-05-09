import numpy as np
import healpy as hp
from astropy.table import Table, vstack
from filter_stars import remove_stars


#Function for healpix to convert from RA/Dec to spherical coordinates. 
def radec_to_sph(ra,dc):
    theta = (90.-dc)*np.pi/180.
    phi   = ra*np.pi/180.
    return theta, phi

#Function to get the healpix pixel numbers for a list of RA and Dec.
def get_n(ra,dec,NSIDE):
    #Convert to ra/dec
    theta, phi = radec_to_sph(ra,dec)
    #set up the healpix grid.
    n = hp.ang2pix(NSIDE, theta, phi)    
    return n


#Main function 
def downselect(out_fname, rlim=21.5, gap_relative_density_definition=0.2, gap_relative_density_target=0.5, area_coords=None, NSIDE=64, star_rejection=True, target_density=None):

    #First we defined the basic healpix parameters we will use. 
    npix = hp.nside2npix(NSIDE)
    total_degrees_in_sky = 4.*np.pi*(180./np.pi)**2
    area_per_pixel = total_degrees_in_sky/(1.*npix)

    #Read the AGN candidate catalogs. 
    Fcat = Table.read("../Victoria_v1.1_SED_catalogs/Master_Catalog_F_test_AGN_full_photometry.fits")
    BIC_cat = Table.read("../Victoria_v1.1_SED_catalogs/Master_Catalog_BIC_AGN_nb_le_6_full_photometry.fits")

    #Apply the magnitude limits.
    Fcat = Fcat[Fcat['mag_auto_r']<rlim]
    BIC_cat = BIC_cat[BIC_cat['mag_auto_r']<rlim]

    #Filter the area requested if provided. 
    if area_coords is not None:
        for k, area_coord in enumerate(area_coords):
            ramin,  ramax  = area_coord[0]
            decmin, decmax = area_coord[1]
            area_cond_F_aux = (Fcat['ra']>ramin) & (Fcat['dec']>decmin)\
                 & (Fcat['ra']<ramax) & (Fcat['dec']<decmax)
            area_cond_B_aux = (BIC_cat['ra']>ramin) & (BIC_cat['dec']>decmin)\
                 & (BIC_cat['ra']<ramax) & (BIC_cat['dec']<decmax)
            if k==0:
                area_cond_F = area_cond_F_aux
                area_cond_B = area_cond_B_aux
            else:
                area_cond_F = area_cond_F | area_cond_F_aux
                area_cond_B = area_cond_B | area_cond_B_aux
        Fcat = Fcat[area_cond_F]
        BIC_cat = BIC_cat[area_cond_B]
        

    #Filter out the stars if requested.
    if star_rejection:
        Fcat = remove_stars(Fcat)
        BIC_cat = remove_stars(BIC_cat)


    #Get the healpix pixel numbers.
    n_F = get_n(Fcat['ra'] , Fcat['dec'] , NSIDE)
    n_B = get_n(BIC_cat['ra'], BIC_cat['dec'], NSIDE)

    #Add up the number of sources per healpix pixel. 
    h_F = np.histogram(n_F,hp.nside2npix(NSIDE), range=(0,hp.nside2npix(NSIDE)-1))[0]
    h_B = np.histogram(n_B,hp.nside2npix(NSIDE), range=(0,hp.nside2npix(NSIDE)-1))[0]

    #If a target source density has not been provided, set it to the median field F-test density.
    if target_density is None:
        median_F_number = np.median(h_F[h_F>0])
        target_density = median_F_number/area_per_pixel
        print("The median density of F-test sources is {:.1f} per sq. deg. Set as target density.".format(target_density))
    else:
        print("The target density of sources is {:.1f} per sq. deg.".format(target_density))

    #This is the histogram that will hold for each healpix pixel the number of sources in the final catalog. 
    h_final = np.zeros(len(h_F))

    #We will create an extra column for each table where we will make notice of whether a given source makes it or not in the final table. 
    Fcat['Final_cat']     = 0.
    BIC_cat['Final_cat']  = 0.

    #This is the main loop to select the sources.
    target_number_per_pix = target_density*area_per_pixel
    gap_limit_number = gap_relative_density_definition*target_number_per_pix
    main_survey_condition = (h_F>=gap_limit_number)
    gap_condition = (h_B>0) & (h_F<gap_limit_number)
    for n in range(npix):

        #If there are no sources, skip to the next step. 
        if h_B[n]+h_F[n]==0:
            continue

        #Set the target number of sources. 
        eff_targ_num = target_number_per_pix
        if gap_condition[n]:
            eff_targ_num = gap_relative_density_target * eff_targ_num
        eff_targ_num = np.int32(eff_targ_num)

        #Make slice conditions. 
        cond_F1 = n_F==n
        cond_B1 = n_B==n

        #Get the number of F-test sources and BIC sources in each pixel. 
        F_num = len(Fcat[cond_F1])
        B_num = len(BIC_cat[cond_B1])

        #If we have enough F-test sources, just use those and keep only the ones with the lowest probability. 
        if F_num>=eff_targ_num:
            Fp_lim = np.sort(Fcat['Fp'][cond_F1])[eff_targ_num-1]
            Fcat['Final_cat'][(cond_F1) & (Fcat['Fp']<=Fp_lim)] = 1

        #If we have enough sources between BIC and F-test, then use all the F-test sources and the best BIC sources. 
        elif F_num + B_num > eff_targ_num:
            Fcat['Final_cat'][cond_F1] = 1
            aux = np.sort(BIC_cat['BIC'][cond_B1])[::-1]
            BIC_lim = aux[eff_targ_num-F_num]
            BIC_cat['Final_cat'][(cond_B1) & (BIC_cat['BIC']>=BIC_lim)] = 1

        #Finally, if not enough between the two of them, just use all sources. 
        else:
            Fcat['Final_cat'][cond_F1] = 1
            BIC_cat['Final_cat'][cond_B1] = 1

    
    #Create a combined downselected table. 
    Fcat_save = Fcat[Fcat['Final_cat']==1]
    BIC_cat_save = BIC_cat[BIC_cat['Final_cat']==1]
    Combined_table = vstack([Fcat_save, BIC_cat_save])

    #Remove the final cat column. 
    Combined_table.remove_column('Final_cat')

    #Add a column with how the source was selected.
    Combined_table['Selection'] = ["F"]*len(Fcat_save['ra']) + ["BIC"]*len(BIC_cat_save)

    #Save the sources.
    Combined_table.write(out_fname, overwrite=True)

