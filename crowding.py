#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Schei008
#
# Created:     04/09/2018
# Copyright:   (c) Schei008 2018
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import geopandas as gpd
import os
import shapely
from shapely import geometry
from shapely.geometry import Point
import matplotlib.pyplot as plt
import random
import time
import math

def param_d(track):
    #Returns an array of distances in a sequence of point geometries in the track
   return [x.distance(track[i[0]+1]) for i,x in np.ndenumerate(track.values) if i[0]+1 < track.values.size]
    #print pd.rolling_apply(track, 2,  lambda x: Point(x[0]["X"],x[0]['Y']).distance(Point(x[1]['X'],x[1]['Y'])))


def Rasterize(track,m):
    lookup = track.apply(lambda p: (np.round(p.x/m)*m, np.round(p.y/m)*m) )
    rt = pd.DataFrame(lookup.unique())
    lookup = lookup.apply(lambda (x,y): Point(x, y) )
    print lookup

    rastertrack = rt.apply(Point)
    print rt
    #rastertrack = gpd.GeoDataFrame(rt, geometry='points')
    #Point(np.round(p.x/m)*m, np.round(p.y/m)*m)

    print("rasterized:")
    print rastertrack
    return lookup,rastertrack




def Crowd(track='data\\301756.csv', k=10, p = 0.1) :
    track['points'] = list(zip(track.X, track.Y))
    track['points']  = track['points'].apply(Point)
    trackgdf = gpd.GeoDataFrame(track, geometry='points')
    trackgdf.crs = {"init": 'epsg:4326'}
    trackgdf = trackgdf.to_crs({"init": 'epsg:28992'})
    #print trackgdf

    #Maximum distance parameter
    distances = param_d(trackgdf['points'])
    d = 1.3* max(distances)
    print('the distance parameter is '+str(d))

    #Rounding increment
    m = math.floor(d/(math.sqrt(2 * k/math.pi)))
    print('the rounding increment is '+str(m))

    #Template radius
    r = math.ceil((-1 + math.sqrt(1+4*k))/2)
    print('the template radius is '+str(r))


    #Start of the programming logic
    lookup,rastertrack = Rasterize(trackgdf['points'],m)








    #print df










def main():
##    track='data\\2420.csv'
##    df = pd.read_csv(track)
##    for track, trackdf in df.groupby("track"):
##        trackdf.to_csv('data\\'+str(track)+".csv")
    df = pd.read_csv('data\\301756.csv')
    track = df[['X','Y']]
    Crowd(track)

if __name__ == '__main__':
    main()
