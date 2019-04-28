#-------------------------------------------------------------------------------
# Name:        Attack obfuscated tracks
# Purpose:     Functions for attacking obfuscated tracks.
#
#
# Author:      Martin Mol, Jiong Wang
#
# Created:     22/10/2018
# Copyright:   (c) Schei008 2018
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas import GeoSeries
import math
import fiona
from rasterstats import zonal_stats, point_query
from fiona.crs import from_epsg
#from shapely import geometry
from sklearn.neighbors import NearestNeighbors
from shapely.geometry import shape, Point, mapping
from scipy import stats
from scipy import spatial
from scipy import signal
### scipy.stats.gaussian_kde


def Smoothing(track, window=3):
    track = np.array([[pt.x, pt.y] for ind, pt in np.ndenumerate(track.values)])
    try:
        def movingaverage(values, window):
            return np.convolve(values, np.repeat(1.0, window)/window, mode='same')
        track = np.array(track)
        track = np.vstack((track[0],track,track[-1]))
        track[:,0] = movingaverage(track[:,0],3)
        track[:,1] = movingaverage(track[:,1],3)
        
        track = np.array(track)
        track = track[1:-1]
        track = GeoSeries(map(Point, track))
        return track
    except ValueError: # If track is shorter than 3 points
        print("Track too short")
        
    return track


def knn(track, k):
    #Table of k-neighbors of each track points
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(track)
    distances, indices = nbrs.kneighbors(track)
    return distances, indices


def getLUProb(loc, land, residential=60, lag=30):
    loc = Point(loc)
    bufferLoc = loc.buffer(lag)
    stats = zonal_stats(bufferLoc, land)
    categoryCount = zonal_stats(bufferLoc, land, categorical=True)
    if residential in stats[0].keys():
        prob = categoryCount[0][residential]/stats[0]['count']
    else:
        prob = 0.0001
    return prob


def centroid(cluster, land, residential):
    density = stats.gaussian_kde(cluster)
    #Places to sample the estimated the kernel density
    xmin = cluster[0].min()
    xmax = cluster[0].max()
    ymin = cluster[1].min()
    ymax = cluster[1].max()
    X, Y = np.mgrid[xmin:xmax:100, ymin:ymax:100]
    positions = np.vstack([X.ravel(), Y.ravel()])
    
    #Joint density of spaital cluster and type
    densityCol = density(positions)  #Density into 1-D for choosing
    densityLoc = []
    for loc in positions.T:
        densityLoc.append(getLUProb(loc, land, residential=60, lag=30))
    densityLoc = np.array(densityLoc)
        
    densityCol = densityCol*densityLoc
##    plotDensity(xmin, xmax, ymin, ymax, cluster, density, positions, X)
    #Randomly choose relocation from the density distribution
    densityCol /= densityCol.sum()  # normalize to 1
    c = positions.T[np.argmax(densityCol)]
    return c


def endPattern(track, land, residential, k=5):    
    track = np.array([[pt.x, pt.y] for ind, pt in np.ndenumerate(track.values)])
    distances, indices = knn(track, k)
    
    #start centroid and end centroid according to distribution of track points
    trackst_c = centroid(track[indices[0]].T, land, residential)
    trackend_c = centroid(track[indices[len(track)-1]].T, land, residential)
    return trackst_c, trackend_c


def attack(track, land, residential):
    # Attacking the entire track
    smoothed = Smoothing(track)
    # Attacking the start and end as potential home locations
    start, end = endPattern(track, land, residential, k=5)
    return smoothed, start, end


def attacking(fake):
#    obfstrack = fake
#    layer = 'sampleLandUse.shp'
#    land = gpd.GeoDataFrame.from_file(layer)
    land = 'landUse.tif'
    
    trackobf = gpd.GeoDataFrame.from_file(fake)
    residential = 60  # to be clarified
    
    smoothed, start, end = attack(trackobf['geometry'], land, residential)
    smoothed.to_file(driver = 'ESRI Shapefile', filename = 'attacked.shp')
    return end
    
    


if __name__ == '__main__':
    main()


