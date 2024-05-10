import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

class HealpyHelper(object):

    def __init__(self, NSIDE):

        #Save input. 
        self.NSIDE = NSIDE

        #Calculate some useful numbers.
        self.npix = hp.nside2npix(NSIDE)
        self.total_degrees_in_sky = 4.*np.pi*(180./np.pi)**2
        self.area_per_pixel = self.total_degrees_in_sky/(1.*self.npix)

        return
    
    def radec_to_sph(self, ra,dc):
        """
        Simple method to convert from RA/Dec to spherical coordinates. Useful for healpix. 

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
    
    def get_n(self,ra,dec):
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
        theta, phi = self.radec_to_sph(ra,dec)
        #set up the healpix grid.
        n = hp.ang2pix(self.NSIDE, theta, phi)    
        return n

    def get_h(self, ra, dec):
        
        #set up the healpix grid.
        n = self.get_n(ra,dec)
        
        #Load up the healpix array and display it. 
        h = np.histogram(n,self.npix,range=(0,self.npix-1))[0]
        h = h.astype(np.float64)
        
        return h
    

    def plot_healpix(self, ra=None,dec=None,fname=None,coord=None,
                 rot=None,title="",cmin=0,cmax=None,h=None,notext=True):

        if rot is None:
            rot = [0,0,0]

        if ra is None and h is None:
            print("You need to provide either a list of coordinates or the h histogramed pixel list")
            return

        if h is None:
            h = self.get_h(ra, dec)

        #Transform into a per degree scale.
        h /= self.area_per_pixel

        #Set the maximum density value for the color
        if cmax is None:
            cmax = np.ceil(np.percentile(h,99.9936)) #4 sigma

        hp.mollview(h,title=title.format(self.NSIDE),rot=rot,coord=coord,
                    notext=notext,min=cmin,max=cmax)
        hp.visufunc.graticule()

        if fname is None:
            plt.show(block=True)
        else:
            plt.savefig(fname, dpi=150)

        return

