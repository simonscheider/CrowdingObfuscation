#-------------------------------------------------------------------------------
# Name:        Evaluating information preservation level of obfuscated track
# Purpose:     Functions for comparing obfuscated tracks with the original track.
#              Main method is Evaluate().
#              Metrics include:
#              1. line pattern based upon track buffer
#              2. Exposure accuracy of obfuscated tracks
#              3. Time efficiency of enrichment
#
# Author:      Jiong Wang
#
# Created:     22/10/2018
# Copyright:   (c) Schei008 2018
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import geopandas as gpd
import math
import georasters as gr

from rasterstats import zonal_stats, point_query
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from shapely import geometry
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from random import randint
from scipy import stats

'''
def gaussianK(size, size_y=None):
    size = int(size)
    if not size_y:
        size_y = size
    else:
        size_y = int(size_y)
    x, y = np.mgrid[-size:size+1, -size_y:size_y+1]
    g = np.exp(-(x**2/float(size)+y**2/float(size_y)))
    return g #/ g.sum()  # Normalize or not


def expose(track, concentration, radius, gaussian=False):
    (xmin, xsize, x, ymax, y, ysize) = concentration.geot
    m = concentration.extract(xmin+int(15*concentration.shape[0]), ymax-int(15*concentration.shape[1]), radius=200)  # central pixel and neighbor matrix
    pred = []
    for i, p in track.iterrows():
        m = concentration.extract(p['geometry'].x, p['geometry'].y, radius)
        w = gaussianK((m.shape[0]-1)/2, (m.shape[1]-1)/2)
        if gaussian:
            pred.append(np.sum(m.raster.data*w)/np.sum(w))
        else:
            pred.append(np.sum(m.raster.data)/(m.shape[0]**2))
    return np.sum(np.array(np.reshape(pred,(-1,1))))
'''

def expo(track, concentration, radius):
    trackLine = geometry.LineString(track)
    #create line buffer
    trackBuffer = trackLine.buffer(radius)
    stats = zonal_stats(trackBuffer, concentration, 
                        stats=['min', 'max', 'mean', 'median', 'sum'])
    return np.array([stats[0]['min'], stats[0]['max'], stats[0]['median'], stats[0]['mean']])  # , stats[0]['sum']/len(track)])

def Evaluate(track, trackobf, concentration, lag=50) :
    expo_obf = expo(trackobf, concentration, radius=5)  # Exposure as concentration within 5 meters of buffer
    expo_real = expo(track, concentration, radius=5)  # Exposure as concentration within 5 meters of buffer
    
    p1 = abs(expo_obf-expo_real)/expo_real
    print('the obfuscation precision on EXPOSURE is '+str(p1))
    return p1



def evalPres(originaltrack, faketrack, concentration):
#    originaltrack = 'track.shp'
#    obfstrack = 'faketrack.shp'
#    concentration = 'no2small.tif'  # Air pollution map as TIFF file (to be created)
    
    track = gpd.GeoDataFrame.from_file(originaltrack)
    trackobf = gpd.GeoDataFrame.from_file(faketrack)
#    concentration = gr.from_file(pollutionmap)  # Air pollution map as distribution concentration
    
    preserve = Evaluate(track['geometry'], trackobf['geometry'], concentration, lag=30)
    return preserve
    
    



if __name__ == '__main__':
    main()

