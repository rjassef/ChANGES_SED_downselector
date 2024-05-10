import numpy as np
import healpy as hp
from astropy.table import Table, vstack
from filter_stars import remove_stars

def radec_to_sph(ra,dc):
    """
    Simple function to convert from RA/Dec to spherical coordinates. Useful for healpix. 

    Parameters
    ----------
    ra  :   numpy array or float
            Value of the R.A. Can be a numpy array.

    dec :   numpy array or float
            Value of the Dec. Can be a numpy array.
    """

    theta = (90.-dc)*np.pi/180.
    phi   = ra*np.pi/180.
    return theta, phi


def get_n(ra,dec,NSIDE):
    """
    Function to get the healpix cell index for a given RA and Dec. 

    Parameters
    ----------
    ra      :   numpy array or float
                Value of the R.A. Can be a numpy array.

    dec     :   numpy array or float
                Value of the Dec. Can be a numpy array.

    NSIDE   :   int
                Healpix NSIDE parameter.

    """
    #Convert to ra/dec
    theta, phi = radec_to_sph(ra,dec)
    #set up the healpix grid.
    n = hp.ang2pix(NSIDE, theta, phi)    
    return n


def downselect(out_fname, rlim=21.5, gap_relative_density_definition=0.2, gap_relative_density_target=0.5, area_coords=None, NSIDE=64, star_rejection=True, target_density=None, Fcat_initial=None, BIC_cat_inital=None):
    """
    This is the main subroutine for downselection of the targets for the SED component of the 4MOST/ChANGES survey. The subroutine divides the sky in healpix cells of a size given by the NSIDE. 
    
    It first read and limits the F-test and BIC catalogs to mag_auto_r < rlim, applies spatial cuts given by the area_coords array, and cuts to eliminates stellar contaminants if star_rejection is True. 

    After that, it creates a catalog with a given source density (if possible) in each healpix cell. First all F-test sources are added, ranked by their F-test probability, and then sources are added from the BIC catalog, ranked by BIC, to try to complete the targeted number of sources per healpix cell (or until all candidates in the cell are exhausted). 

    If no target source density is provided (target_density parameter), the target_density is automatically set to the median source density of the healpix cells of the F-test catalog (after magnitude, spatial, an stellar color cuts are applied) within the region of interest. 

    Healpix cells where the F-test source density falls below a fraction of the targeted density (given by the gap_relative_density_definition parameter) are only filled up to a source density given by gap_relative_density_target * target_density.

    Parameters
    ----------

    out_fname                       :   str
                                        Name of the output catalog file. Existing files will be overwritten.
    
    rlim                            :   float
                                        auto_mag_r magnitude limit of the output catalog. Default is r=21.5.

    gap_relative_density_definition :   float
                                        Relative density with respect to the target density of a healpix cell that defines it as a gap region (i.e., region of lower depth or band coverage). Default is 0.2. 
    
    gap_relative_density_target     :   float
                                        Healpix cells defined as gap cells are only filled up to this relative fraction of the target_density. Default value is 0.5.
    
    area_coords                     :   list or float array of dimension Nx2x2
                                        Spatial cuts to target specific regions. Multiple regions can be defined, each corresponding to the first index of the array or list. The second index holds the minimum and maximum RAs of the region in the first position and the minimum and maximum Decs in the second position. Default value is None.
    
    NSIDE                           :   int
                                        Healpix NSIDE parameter. Determines the cell size. Default value is 64.
    
    star_rejection                  :   bool
                                        Boolean that determines whether to apply stellar rejection color cuts or not. Default value is True.
    
    target_density                  :   float
                                        Targeted source density for each healpix cell. If give a value of None, the target density is set to be the median density of the healpix cells of the F-test catalog (after magnitude, spatial and stellar rejection cuts). Default value is None.

    Fcat_initial                    :   astropy.table
                                        Initial table of F-test candidates to which we start applying the cuts. If None, the default table is loaded. 

    BIC_cat_initial                 :   astropy.table
                                        Initial table of BIC candidates to which we start applying the cuts. If None, the default table is loaded.

    """

    #First we defined the basic healpix parameters we will use. 
    npix = hp.nside2npix(NSIDE)
    total_degrees_in_sky = 4.*np.pi*(180./np.pi)**2
    area_per_pixel = total_degrees_in_sky/(1.*npix)

    #Read the AGN candidate catalogs. 
    if Fcat_initial is None:
        Fcat = Table.read("../Victoria_v1.1_SED_catalogs/Master_Catalog_F_test_AGN_full_photometry.fits")
    else:
        Fcat = Fcat_initial
    
    if BIC_cat_inital is None:
        BIC_cat = Table.read("../Victoria_v1.1_SED_catalogs/Master_Catalog_BIC_AGN_nb_le_6_full_photometry.fits")
    else:
        BIC_cat = BIC_cat_inital

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

