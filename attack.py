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


def centroid(cluster):
    density = stats.gaussian_kde(cluster)
    #Places to sample the estimated the kernel density
    xmin = cluster[0].min()
    xmax = cluster[0].max()
    ymin = cluster[1].min()
    ymax = cluster[1].max()
    X, Y = np.mgrid[xmin:xmax:2, ymin:ymax:2]
    positions = np.vstack([X.ravel(), Y.ravel()])
##    plotDensity(xmin, xmax, ymin, ymax, cluster, density, positions, X)
    #Randomly choose relocation from the density distribution
    densityCol = density(positions)  #Density into 1-D for choosing
    densityCol /= densityCol.sum()
    c = positions.T[np.argmax(densityCol)]
    return c


def endPattern(track, k=10):    
    track = np.array([[pt.x, pt.y] for ind, pt in np.ndenumerate(track.values)])
    distances, indices = knn(track, k)
    
    #start centroid and end centroid according to distribution of track points
    trackst_c = centroid(track[indices[0]].T)
    trackend_c = centroid(track[indices[len(track)-1]].T)
        
    return trackst_c, trackend_c


def pickHome(track, land, residential, lag=30):
    trackPt = [Point(float(track[i[0]].x),float(track[i[0]].y)) for i,p in np.ndenumerate(track.values) if i[0]+1 < track.values.size]
    #use a buffer to summarize land use specification at each existing location
    buffer = [point.buffer(lag) for point in trackPt]
    ##merged = unary_union(buffer)
    schemaBuffer = {'geometry': 'Polygon','properties': {'Id':'int:5'},}
    with fiona.open("buffer.shp", "w", driver='ESRI Shapefile', crs=from_epsg(28992), schema=schemaBuffer) as output:
        for b in buffer:
            output.write({'geometry': mapping(b), 'properties': {'Id':buffer.index(b)}})

    location = gpd.GeoDataFrame.from_file("buffer.shp")
    data = []
    for index, sample in location.iterrows():
        for index2, parcel in land.iterrows():
            if sample['geometry'].intersects(parcel['geometry']):
                data.append({'geometry': sample['geometry'].intersection(parcel['geometry']), 'location':sample['Id'], 'land': parcel['BG2010'], 'area':sample['geometry'].intersection(parcel['geometry']).area})
                df = gpd.GeoDataFrame(data,columns=['geometry', 'location', 'land','area'])
                df.to_file('intersection.shp')
                # control of the results in mi case, first values
                df.head() # image from a Jupiter/IPython notebook

    bufferLand = gpd.GeoDataFrame.from_file("intersection.shp")
    bufferAgg = bufferLand[['geometry', 'location', 'area']]
    bufferAgg = bufferAgg.dissolve(by='location', aggfunc='max', as_index=False)
    ##bufferAgg.tail(2)

    bufferAgg['land'] = None
    bufferAgg['localProb'] = None
    bufferAgg['overallProb'] = None
    landOverall = []
    for index, attr in bufferAgg.iterrows():
        for index2, attr2 in bufferLand.iterrows():
            if attr['area'] == attr2['area']:
                bufferAgg['land'][index] = attr2['land']
                bufferAgg['localProb'][index] = attr['area']/float(math.pi*lag**2)
                ## land type probability along the entire track
                landOverall = [bufferLand['area'][index3] for index3, attr3 in bufferLand.iterrows() if attr3['land'] == attr2['land']]
                bufferAgg['overallProb'][index] = sum(landOverall)/float(len(bufferAgg)*math.pi*lag**2)
                df = gpd.GeoDataFrame(bufferAgg,columns=['geometry', 'location', 'land', 'area', 'localProb', 'overallProb'])
                df.to_file('bufferAgg.shp')
                df.head()
    
    potentialhome = []
    for index, attr in bufferAgg.iterrows():
        if attr['land'] == residential:
            potentialhome.append(bufferAgg['geometry'][index])

    return potentialhome


def attack(track, land, residential):
    # Attacking the entire track
    smoothed = Smoothing(track)
    # Attacking the start and end as potential home locations
    start, end = endPattern(track, k=10)
    # Attacking the home location through land use
    #homelocations = pickHome(track, land, residential)
    
    return smoothed, start, end


def attacking():
    obfstrack = 'faketrack.shp'
#    layer = 'sampleLandUse.shp'
#    land = gpd.GeoDataFrame.from_file(layer)
    land = 'landUse.tif'
    
    trackobf = gpd.GeoDataFrame.from_file(obfstrack)
    residential = 60  # to be clarified
    
    smoothed, start, end = attack(trackobf['geometry'], land, residential)
    smoothed.to_file(driver = 'ESRI Shapefile', filename = 'attacked.shp')
    
    


if __name__ == '__main__':
    main()


