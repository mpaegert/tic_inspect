#!/usr/bin/env python
'''
@package : tic_contam
@author  : map
@version : \$Revision$
@Date    : \$Date$

Description

Compute contamination ratio ( = contaminating flux / target flux) and its error
for TIC8 stars from public database at MAST. Stars up to 10 TESS pixel away are 
taken into account.

Requires python3, scipy, the astropy suite, astroquery (MAST).
    tic_contam.py --help
prints a help-message. Either --ticid or --ticidfile must be set.

The result for
    tic_contam.py --ticid=46
should be:
   TICID   !  Tmag ! e_Tmag! contam   ! e_contam ! nrcont
        46 ! 14.26 !  0.01 ! 0.266572 ! 0.002773 !   172

the logfile (default logfile.txt contains the formatted output, fluxes.csv 
(default name) contains the output as csv for easy processing.
 
$Log$
Initial revision
'''

import sys
import csv

from math import sqrt, log, pi, sin, cos
from pprint import pprint
from copy import deepcopy
from optparse import OptionParser

import scipy.special as npspec

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, ICRS, FK4, Angle

from astroquery.mast import Catalogs
import numpy as np


ticepoch    = Time('J2000.0')
options = None
lf      = None

tess_pixel = 20.25                 # arcsec
tess_scale = tess_pixel / 3600.0   # arcsec / pixel --> deg / pixel
tess_fwhm  = 1.88  * tess_scale    # new fwhm from camera image (sigma = 0.80)
tess_aper  = 4.0  * tess_scale     # 4 pixel aperture
tess_srad  = 10.0 * tess_scale     # 10 pixel search radius 
tess_sigma = tess_fwhm / (2.0 * sqrt(2.0 * log(2)))
deg2rad    = pi / 180.0
rad2deg = 1 / deg2rad

csvcols = ['ticid', 'tessmag', 'tessmage', 'contamratio', 'contamratioe', 'nrcontam']
lfmt = ('{:>10s} ! {:5.2f} ! {:5.2f} ! {:8.6f} ! {:8.6f} ! {:5d}')
head = '   TICID   !  Tmag ! e_Tmag! contam   ! e_contam ! nrcont'


def nrw(whereresult):
    return np.shape(whereresult)[1]



def get_ticlist(ticfile):
    ticids = []
    with open(ticfile) as tf:
        for line in tf:
            line = line.strip('\n')
            if len(line) == 0 or line.startswith('#'):
                continue
            tmp = line.split(',')
            for id in tmp:
                tid = id.strip()
                if len(tid) == 0:
                    continue
                ticids.append(int(tid))
    
    return ticids



def propagate_pm(ticstars):
    ''' propagate proper motions to target epoch, re-compute the radial
        distances and sort the array by distances again
    '''
    nrstars = len(ticstars)
    epochs  = np.ndarray(nrstars, dtype = 'object')
    epochs.fill(ticepoch)
    mask = ~np.isnan(ticstars['pmRA'].data.data) & ~np.isnan(ticstars['pmDEC'].data.data)
    gd = np.where(mask)
    ngd = nrw(gd)
    if ngd == 0:
        # nothing to do
        return
    
    # assume 1000 pc distance for stars without known distance
    dist = ticstars['d']
    bd = np.where(np.isnan(dist.data.data))
    dist[bd] = 1000.0

    # setup astropy coordinates, apply space motion and correct coordinates   
    coords = SkyCoord(ra = ticstars['ra'][gd] * u.deg,
                      dec = ticstars['dec'][gd] * u.deg, 
                      pm_ra_cosdec = ticstars['pmRA'][gd] * u.mas / u.yr, 
                      pm_dec = ticstars['pmDEC'][gd] * u.mas / u.yr, 
                      obstime = ticepoch, 
                      distance = dist[gd] * u.pc)
    newcoords = coords.apply_space_motion(Time(options.targetepoch))
    newra  = newcoords.ra.degree
    newdec = newcoords.dec.degree
    oldra  = deepcopy(ticstars['ra'])
    olddec = deepcopy(ticstars['dec'])
    ticstars['ra'][gd] = newra
    ticstars['dec'][gd] = newdec
    
    # recompute the angular distances with new coordinates
    c0 = SkyCoord(ra = ticstars['ra'][0] * u.deg, 
                  dec = ticstars['dec'][0] * u.deg)
    for i in range(1, nrstars):     # skip the target star itself
        cs = SkyCoord(ra = ticstars['ra'][i] * u.deg, 
                      dec = ticstars['dec'][i] * u.deg)
        rdist = c0.separation(cs)
        ticstars['dstArcSec'][i] = rdist.arcsecond
    
    # resort table by new radial distance
    ticstars.sort(['dstArcSec'])



def aperture(tmag, tmage):
    ''' estimate the TESS aperture for given TESS magnitude
    '''
    
    if tmage is None:
        tmage = 0.25
    
    npix  = None
    npixe = None

    if tmag <= 4.0:
        npix  = 274.2898 - 77.7918 * 4.0 + 7.7410 * 4.0**2 - 0.2592 * 4.0**3
        npixe = abs((77.7918 + 2 * 7.7410 * 4.0 - 3 * 0.2592 * 4.0**2) * tmage)
    else:
        npix  = (274.2898 - 77.7918 * tmag + 
                 7.7410 * tmag**2 - 0.2592 * tmag**3)
        npixe = abs(77.7918 + 2 * 7.7410 * tmag - 3 * 0.2592 * tmag**2) * tmage
        if npix < 1:
            npix  = 1
            npixe = 0.05
    
    # square of the same area - this will underestimate the contaminating flux
    aper1  = sqrt(npix)
    apere1 = abs(0.5 / sqrt(npix) * npixe)
    
    # surrounding square - area by factor 4 / pi bigger than circle
    # this will overestimate the contaminating flux
    aper2  = 2 * sqrt(npix / pi)
    apere2 = 2 * sqrt(1.0 / (npix * pi)) * npixe
    
    # average over surrounding square and square of the same area
    # this should give a more realistic estimate
    aper   = (aper1 + aper2) / 2
    apere  = (apere1 + apere2) / 2
    
    return (aper, apere)



def tmag2flux(tmag, tmagerr = None):
    # use internal functions for single values and numpy arrays, providing
    # the same call signature no matter if called with scalars or arrays
    if (tmag is None):
        return (None, None)
    flux = 2635.0 * pow(10.0, -tmag / 2.5)
    ferr = None
    if (tmagerr is not None):
        ferr = abs(-log(10) / 2.5 * flux * tmagerr)
    return (flux, ferr) 



def haversine(ra1, dec1, ra2, dec2,
              dra1 = None, ddec1 = None, dra2 = None, ddec2 = None):
    ras  = ra1 * deg2rad
    raf  = ra2 * deg2rad
    decs = dec1 * deg2rad
    decf = dec2 * deg2rad

    dra  = raf - ras
    ddec = decf - decs
    tmp1 = np.sin(ddec / 2)**2
    tmp2 = np.cos(decs) * np.cos(decf) * np.sin(dra / 2)**2
    delta = 2.0 * np.arcsin(np.sqrt(tmp1 + tmp2))
    delta = delta * rad2deg
    
    ddelta = None
    if (dra1 is not None or ddec1 is not None or 
        dra2 is not None or ddec2 is not None):
        tmp   = sin(0.5 * (dec2 - dec1))**2 * cos(ra1) * cos(ra2) + \
                sin(0.5 * (ra2 - ra1))**2
        denom = 4 * (1 - tmp) * tmp
        ddelta = np.array(len(ra1) * [0.0])
        if dra1 is not None:
            ddelta += (np.cos(dec1 - dec2) * np.cos(ra2) * np.sin(ra1) - 
                       np.cos(ra1) * np.sin(ra2))**2 / denom * (dra1 * deg2rad)**2
        if dra2 is not None:
            ddelta += (np.sin(ra1 - ra2) + 
                       2 * np.cos(ra1) * np.sin(ra2) * 
                       np.sin(0.5 * (dec2 - dec1))**2)**2 / \
                         denom * (dra2 * deg2rad)**2
        if ddec1 is not None:
            ddelta += (np.cos(ra1) * np.cos(ra2) * np.sin(dec1 - dec2))**2 / \
                      denom * (ddec1 * deg2rad)**2
        if ddec2 is not None:
            ddelta += (np.cos(ra1) * np.cos(ra2) * np.sin(dec1 - dec2))**2 / \
                      denom * (ddec2 * deg2rad)**2
        ddelta = np.sqrt(ddelta) * rad2deg
    
    return (delta, ddelta)



def contam(x0, y0, xb, yb, s, dx0 = None, dy0 = None):
    ''' for the distances x0, y0 and an aperture xb, yb it returns the fraction
        of the flux for each object at this distance that falls into the aperture.
        The function assumes a Gaussian with s as sigma.
    '''
    sq2 = sqrt(2)
    contx = npspec.erf((xb + x0) / (sq2 * s)) + npspec.erf((xb - x0) / (sq2 * s))
    conty = npspec.erf((yb + y0) / (sq2 * s)) + npspec.erf((yb - y0) / (sq2 * s))
    cont  = 0.25 * contx * conty

    dcont = np.zeros(len(cont))
    if dx0 is not None or dy0 is not None:
#             dcont = 0.0
        tmp = 0.25 * sqrt(2.0 / pi) / s
        dnm = 2.0 * s * s
        if dx0 is not None:
            dcont += (tmp * (np.exp(-(xb + x0)**2 / dnm) - np.exp(-(xb - x0)**2 / dnm)) * 
                      conty * dx0)**2
        if dy0 is not None:
            dcont += (tmp * (np.exp(-(yb + y0)**2 / dnm) - np.exp(-(yb - y0)**2 / dnm)) * 
                      contx * dy0)**2
        dcont = np.sqrt(dcont)
    
    return (cont, dcont)



if __name__ == '__main__':
    print('command line args =', sys.argv)
    usage = '\n%prog [options] - ticid or ticfile option must be set'
    parser = OptionParser(usage = usage)
    parser.add_option('--csvfile', dest='csvfile', type='string', default='fluxes.csv',
                      help='logfile for analysis (default = fluxes.cvs)')
    parser.add_option('--debug', dest='debug', type='int', default=0,
                      help='debug level (default = 0)')
    parser.add_option('--logfile', dest='logfile', type='string', default='logfile.txt',
                      help='logfile for analysis (default = report.txt)')
    parser.add_option('--ticidfile', dest='ticfile', type='string', default=None,
                      help='file with ticids (default = None)')
    parser.add_option('--ticid', dest='ticid', type='int', default=None,
                      help='ticid for the target (default = None)')
    parser.add_option('--tepoch', dest='targetepoch', type='string', 
                      default='J2019.5',
                      help='target epoch for proper motions (default = J2019.5)')

    (options, args) = parser.parse_args()
    
    # options.ticid = 46
    # options.ticid = 2757293
    if options.ticid is None and options.ticfile is None:
        parser.print_help()
        exit(1)
        
    if options.ticid:
        options.ticid = [options.ticid]   # make it a list
    if options.ticfile:
        options.ticid = get_ticlist(options.ticfile)

    if options.logfile is not None:
        lf = open(options.logfile, 'w')

    csvf    = open(options.csvfile, 'w')
    csvfile = csv.writer(csvf, delimiter = ',')
    csvfile.writerow(csvcols)
    print(head)
    lf.write(head + '\n')

    for actid in options.ticid:
        target_name = 'TIC ' + str(actid)
        ticid       = target_name[4:]
        # target_name = '330.794887332661, 18.8843189579296'

        # Query the TESS Input Catalog centered on the target_name. 
        # target_name will be resolved by Simbad, so we need 'TIC ' in front of
        # the id. catallog = 'TIC' is for the MAST radial query.
        ticstars = Catalogs.query_object(target_name, radius = tess_srad, 
                                         catalog = 'TIC')

        # What columns are available from the TIC?
        # print('available columns =', ticstars.columns)
        # print(len(ticstars), 'stars found')

        # propagate proper motions
        propagate_pm(ticstars)

        # get a copy of the target star itself, then delete if from the list 
        where_self = np.where(ticstars['ID'] == ticid)[0]
        target = deepcopy(ticstars[where_self[0]])
        del ticstars[where_self[0]]
        nrcontam = len(ticstars)
        tmag  = target['Tmag']
        if np.isnan(target['e_Tmag']):
            tmage = 0.25
        else:
            tmage = target['e_Tmag']
        contratio   = 0.0
        contratio_e = 0.0
        
        if nrcontam == 0:
            cols = [ticid, tmag, tmage, contratio, contratio_e, nrcontam]
            csvfile.writerow(cols)
            continue
        
        (aper, apere) = aperture(tmag, tmage)
        aper  = aper * tess_scale
        apere = apere * tess_scale

        try:
            (tflux, tfluxe) = tmag2flux(tmag, tmage)
            if options.debug > 1:
                print('tflux, tfluxe = ', tflux, tfluxe, ' (target flux and error)')
        except Exception:
            e, v = sys.exc_info()[:2]
            lf.write('\n')
            lf.write('while computing tflux for ticid ' + str(ticid) + '\n')
            lf.write('ERROR: ' + str(e) + '\n')
            lf.write('VALUE: ' + str(v) + '\n')
            raise

        if not tflux:
            msg = 'no flux for ticid ' + str(ticid) + '\n'
            lf.write(msg)
            raise Exception(msg)
        
        # if we don't have an error, assume 0.25 mag
        bd = np.where(np.isnan(ticstars['e_Tmag']))
        ticstars['e_Tmag'][bd] = 0.25
        
        # get the contaminating fluxes
        (cflux, cfluxe) = tmag2flux(ticstars['Tmag'], ticstars['e_Tmag'])
        
        # compile information for contamfluxes, we need distances in ra and dec
        # for contam
        (dra, _drae)   = haversine(target['ra'], target['dec'], 
                                   ticstars['ra'], target['dec'])
        (ddec, _ddece) = haversine(target['ra'], target['dec'], 
                                   target['ra'], ticstars['dec'])
        (tmp, _dtmp)   = contam(dra, ddec, aper / 2.0, aper / 2.0, tess_sigma)
        cflx = tmp * cflux
        cfle = abs(tmp * cfluxe)

        # sum it up
        sumcflx = np.sum(cflx)
        sumcfle = np.sum(cfle)
        if np.isnan(sumcflx):
            msg = 'sumcflx is nan for ticid %d \n' % (ticid,) 
            lf.write(msg)

        # get the ratio of contaminating flux to target flux
        flxratio  = sumcflx / tflux
        flxratioe = sqrt((sumcfle / tflux)**2 + (sumcflx * tfluxe / tflux**2)**2)

        cols = [ticid, tmag, tmage, flxratio, flxratioe, nrcontam]
        csvfile.writerow(cols)
        msg = lfmt.format(ticid, tmag, tmage, flxratio, flxratioe, nrcontam)
        lf.write(msg + '\n')
        print(msg)

    print('done')
