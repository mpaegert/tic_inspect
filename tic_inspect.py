#!/usr/bin/env python
'''
@package : toi_inspect
@author  : map
@version : \$Revision$
@Date    : \$Date$

Description

Inspect neighbourhood of TICIDs for potential artifacts, joins and splits. 
Requires python3, the astropy suite and astroquery.
       tic_inspect.py --help
prints a help-message. Either --ticid or --ticidfile must be set.
       tic_inspect.py --ticidfile=ticidfile.txt
will run the sample ticids and produce a report in 'report.txt' and a 
comma separated list in phantoms.csv. Please note that any ticid_target in the 
csv-file with a DUPLICATE plus at least one other line with only a duplicate_id 
is a SPLIT and should be treated as such.

sample artifact around ticid 269701147, artifact ticid = 269701145
sample join:  ticid 76989773 with ticid 2055898683
sample split: ticid 13419950 into ticid 1969293164 and 1969293163

this is trouble: join ticid = 430528566, join with 2014876461, there is a 
second 2MASS star close by because it seems its right on the border of two
2MASS images. In addition its right in the plane (glat = -2.8 deg)

this is trouble: ticid 470315428 has 4 other ticids within 5.0 asec. At least
3 of them are bona-fide stars with are only in Gaia but not in 2MASS. This is 
right in the plane again: glat = 1.8 deg.

$Log$
Initial revision
'''

import csv

from pprint import pprint
from copy import deepcopy
from optparse import OptionParser

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, ICRS, FK4, Angle

from astroquery.mast import Catalogs
import numpy as np


tess_pixel = 20.25    # asec

ticepoch    = Time('J2000.0')
targetepoch = Time('J2019.5')

tfmt = '{:>10s} ! {:7.3f} ! {:19s} ! {:16s} !  {}{}{} ! {:11s} ! {} \n\n'
lfmt = ('{:9d} ! {:10d} ! {:19s} ! {:4s} ! {:5.2f} ! {:8.4f} ! {:8.4f} ! ' +
        '{:8.4f}   ! {:11s} ! {}')
lf      = None
options = None
target  = None
nrarti  = 0

csvcols = ['ticid_target', 'ticid_phantom', 'tmflags_phantom', 'radial_dist',
           'tessmag_target', 'tessmag_phantom', 'target_phantom',
           'disposition_phantom', 'dupeid_phantom']



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



def report_phantom(phantom, disposition, dupeid):
    gaiapk = str(phantom['GAIA'])
    qflag  = str(phantom['TWOMflag'])
    if len(qflag) > 3:
        qflag = qflag[:3]
    elif qflag == '--':
        qflag = '---'
    msg = lfmt.format(int(target['ID']), int(phantom['ID']),
                      gaiapk, 
                      qflag[:3], 
                      round(phantom['dstArcSec'], 2), 
                      target['Tmag'], phantom['Tmag'],
                      round(target['Tmag'] - phantom['Tmag'], 4),
                      disposition, dupeid
                      )
    print(msg)
    if lf:
        lf.write(msg + '\n')

    gaiapk = gaiapk.strip('-')
    qflag  = qflag.strip('-')
    cols = [target['ID'], phantom['ID'], 
            gaiapk, 
            qflag[:3],
            round(phantom['dstArcSec'], 2), 
            target['Tmag'], phantom['Tmag'],
            target['Tmag'] - phantom['Tmag'],
            disposition, dupeid]
    csvfile.writerow(cols)



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
    newcoords = coords.apply_space_motion(targetepoch)
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

    

def write_target(target):
    msg  = 'Target\n' 
    msg += ('TICID      ! tessmag ! gaiapk              ! 2MASS            ! qual !'
            ' disposition ! duplicate_id \n')
    gaiapk = str(target['GAIA'])
    disp   = str(target['disposition'])
    dupeid = str(target['duplicate_id'])
    msg += tfmt.format(target['ID'], target['Tmag'], gaiapk,
                       target['TWOMASS'], 
                       target['TWOMflag'][0], target['TWOMflag'][1], 
                       target['TWOMflag'][2], 
                       disp, dupeid)
    print(msg, end = '')
    if lf:
        lf.write(msg)

    # Print out the number of returned rows.
    msg = ('%d objects within %f deg of %s \n' % 
           (len(ticstars), 3600 * search_radius_deg, target_name))
    print(msg, end = '')
    if lf:
        lf.write(msg)
        lf.write('\n\n')

    print('ra, dec = ', target['ra'], target['dec'])


def write_artifacts(ticstars):

    artifacts  = []
    nrstars = len(ticstars)
    for i in range(nrstars):
        if not ticstars['GAIA'].mask[i] or ticstars['TWOMASS'].mask[i]:
            # stars with a Gaiaid are no artifacts, neither are non-2mass stars
            continue
        if not ticstars['disposition'].mask[i] or not ticstars['duplicate_id'].mask[i]:
            # stars with known disposition or dupeid are no new artifacts,
            # they have been dealt with already
            continue
        qual = str(ticstars['TWOMflag'][i])
        if len(qual) < 3:
            qual = '---'

        isartifact = False
        cnt = 0            
        for j in range(3):
            if qual[j] == 'U':
                cnt += 1
        if cnt >= 2:
            isartifact = True
        if isartifact == False:
            continue
        artifacts.append(i)
        
        report_phantom(ticstars[i], 'ARTIFACT', '')
        
    return artifacts



def joins_splits(ticstars):
    gd = np.where((~ticstars['GAIA'].mask) & (ticstars['TWOMASS'].mask))[0]
    for i, idx in enumerate(gd):
        if i == 0:
            report_phantom(ticstars[idx], 'DUPLICATE', target['ID'])
        else:
            report_phantom(ticstars[idx], '', target['ID'])
    
    if lf:
        if len(gd) > 1:
            lf.write('\n')
            lf.write('Set disposition for ticid ' + target['ID'] + ' to SPLIT\n')
        lf.write('\n')
        lf.write('done with ticid ' + str(ticid) + '\n')            
        lf.write('\n-------------------------------\n\n')
        
    return



if __name__ == '__main__':
    
    usage = '%prog [options] textfile(s)'
    parser = OptionParser(usage = usage)
    parser.add_option('--csvfile', dest='csvfile', type='string', default='phantoms.csv',
                      help='logfile for analysis (default = phantoms.cvs)')
    parser.add_option('--logfile', dest='logfile', type='string', default='report.txt',
                      help='logfile for analysis (default = report.txt)')
    parser.add_option('--ticidfile', dest='ticfile', type='string', default=None,
                      help='file with ticids (default = None)')
    parser.add_option('--ticid', dest='ticid', type='int', default=None,
                      help='ticid for the target (default = None)')
    parser.add_option('--artrad', dest='artrad', type='float', default=tess_pixel,
                      help='artifact search radius in asec (default = 20.25)')
    parser.add_option('--joinrad', dest='joinrad', type='float', default=5.0,
                      help='search radius in asec for joins/splits (default = 5.0)')
    
    (options, args) = parser.parse_args()
    
#     options.ticid = 269701147  # artifact
#     options.ticid = 76989773   # join
#     options.ticid = 13419950   # split
#     options.ticid = 141776043  # multiple with one additional bona-fide star
#     options.ticid = 470315428  # multiple with 2 additional bona-fide stars
    options.ticid = 320525204
    if options.ticid is None and options.ticidfile is None:
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

    for id in options.ticid:
        target_name = 'TIC ' + str(id)
        ticid       = target_name[4:]
        # target_name = '330.794887332661, 18.8843189579296'
        search_radius_deg = options.artrad / 3600.0
        
        # Query the TESS Input Catalog centered on the target_name. 
        # target_name will be resolved by Simbad, so we need 'TIC ' in front of
        # the id. catallog = 'TIC' is for the MAST radial query.
        ticstars = Catalogs.query_object(target_name, radius = search_radius_deg, 
                                         catalog = 'TIC')
        
        # What columns are available from the TIC?
        # print(len(ticstars), 'stars found')
        # print(ticstars.columns)
        
        # propagate proper motions
        propagate_pm(ticstars)
        
        # get a copy of the target star itself, then delete if from the list 
        where_self = np.where(ticstars['ID'] == ticid)[0]
        target = deepcopy(ticstars[where_self[0]])
        del ticstars[where_self[0]]
        nrstars = len(ticstars)
            
        write_target(target)
    
        if lf:
            lf.write('target    ! phantom    ! phantom_gaiapk      ! qual ! rdist ! ' + 
                     'targtmag ! phantmag ! d(tessmag) ! disposition ! duplicate_id\n')
        artifacts = write_artifacts(ticstars)
        nartifacts = len(artifacts)
        if nartifacts == nrstars:
            # nothing more to do
            if lf:
                lf.write('\n')
                lf.write('No more stars left within ' + str(options.joinrad) + ' asec\n')
                lf.write('done with ticid ' + str(ticid) + '\n')
                lf.write('\n-------------------------------\n\n')
            continue
        
        for i in range(nartifacts - 1, -1, -1):
            del ticstars[i]
        
        nremain = len(ticstars)
        bad = np.where(ticstars['dstArcSec'] > options.joinrad)[0]
        del ticstars[bad]
        nremain = len(ticstars)
        if nremain == 0:
            # nothing more to do
            if lf:
                lf.write('\n')
                lf.write('No more stars left within ' + str(options.joinrad) + ' asec\n')
                lf.write('done with ticid ' + str(ticid) + '\n')            
                lf.write('\n-------------------------------\n\n')
            continue
        
        if target['GAIA']:
            msg = 'TICID ' + str(target['ID']) + ' has a GAIA ID and thus is no '
            msg += 'candidate for a join or split\n'
            print(msg, end = '')
            if lf:
                lf.write(msg)
                lf.write('done with ticid ' + str(ticid) + '\n')            
                lf.write('\n-------------------------------\n\n')
            continue
            
        print('finding joins and splits in', nremain, 'remaining stars')
    
        joins_splits(ticstars)
    
    csvf.close()
    if lf:
        lf.write('\n')
        lf.write(str(len(options.ticid)) + ' stars inspected\n')
        lf.write('done\n')
        lf.close()
    
    print('done')
