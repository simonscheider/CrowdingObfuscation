#-------------------------------------------------------------------------------
# Name:        Crowding
# Purpose:     Functions for implementing a crowding obfuscation strategy for point tracks.
#              Each track is handled in terms of a geopandas data frame with shapely geometries.
#               Main method is Crowd().
#
# Author:      Simon Scheider
#
# Created:     04/09/2018
# Copyright:   (c) Schei008 2018
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import geopandas as gpd
import fiona
import operator
from fiona.crs import from_epsg
import os
import shapely
from shapely import geometry
from shapely.geometry import shape, Point, mapping
from shapely.ops import unary_union
from rasterstats import zonal_stats, point_query
import random
import time
import math
import sys
from random import randint
import queue as Q
from pathlib import Path


import georasters as gr
#see https://github.com/ozak/georasters
import pyproj
#see https://github.com/jswhit/pyproj
from shapely.ops import transform
from scipy import stats
import rtree

#Spatial vector operations. These are used to implement vector algebra with shapely geometries for template masking
def Vminus(p1, p2):  #Vector subtraction
    return Point(p1.x - p2.x, p1.y - p2.y)

def Vplus(p1, p2):  #Vector addition
    return Point(p1.x + p2.x, p1.y + p2.y)

def Vmult(m, p): #Multiplication of a vector with a scalar
    return Point(p.x*m, p.y*m)


"""Functions for rasterization of track"""

def Rasterize(track,m):
    print('Size of original track:'+str(track.size))
    #This rounds each point in the track based on rounding increment m
    lookup = track.apply(lambda p: (np.round(p.x/m)*m, np.round(p.y/m)*m) )
    rt = pd.DataFrame({'points': lookup.unique()})
    print("lookup table (for picking enrichments for each point of the initial track):")
    lookup = lookup.apply(lambda points: Point(points))
    print(lookup)

    rt['geom'] = rt['points'].apply(Point)

    rastertrack = gpd.GeoDataFrame(rt, geometry='geom')['geom']

    print("rasterized track:")
    print(rastertrack)

    return lookup,rastertrack



"""Functions for track extension (mimick)"""

#Generates a list of difference vectors for a track sequence. Also generates a corresponding set  with unique point values.
def getV(track):
    #Get the vector difference between all pairs of points in the track
     V = [Vminus(track[i[0]+1], p) for i,p in np.ndenumerate(track.values) if i[0]+1 < track.size]
     Vset =[]
     for v in V:
        #count = 0
        insert = True
        for vv in Vset:
            if not v.equals(vv):
                insert = True
            else:
                insert = False
                break
        if insert:
            Vset.append(v)
     return  V, Vset

#Generates a distance list for a track sequence
def getDistances(track):
    #Returns an array of distances in a sequence of point geometries in the track
   return [x.distance(track[i[0]+1]) for i,x in np.ndenumerate(track.values) if i[0]+1 < track.values.size]


"""A function for looking up Raster cell row/colum projected in RD_new, based on a WGS84 coordinate pair as input as well as a Geo raster tile"""
#project = lambda x, y: pyproj.transform(pyproj.Proj(init='EPSG:4326'), pyproj.Proj(init='EPSG:28992'), x, y)
##def lookupWGS84(x,y,GeoT):
##    pass
##    p = shapely.geometry.point.Point(x,y)
##    print(str(p))
##    pn = transform(project, p)
##    print(str(pn))
##    try:
##        # Find location of point (x,y) on raster, e.g. to extract info at that location
##        col, row = gr.map_pixel(pn.x,pn.y,GeoT[1],GeoT[-1], GeoT[0],GeoT[3])
##        #value = gr.map_pixel(pn.x,pn.y, )
##    except:
##        print "Coordinates out of bounds!"
##        return 'NN','NN'
##    return row, col

#see https://blog.maptiks.com/spatial-queries-in-python/
'''
def generate_index(records, index_path):
    prop = rtree.index.Property()
    if index_path is not None:
        prop.storage = rtree.index.RT_Disk
        prop.overwrite = index_path

    sp_index = rtree.index.Index(index_path, properties=prop)
    for n, ft in enumerate(records):
        if ft['geometry'] is not None:
            sp_index.insert(int(n), shape(ft['geometry']).bounds)
    return sp_index
'''


def getLUProb(newLoc, land, table, lag):
    buffernewLoc = newLoc.buffer(lag)
    stats = zonal_stats(buffernewLoc, land, categorical=True)
    # Most frequent land use type as the mode
    modeLand = max(stats[0].items(), key=operator.itemgetter(1))[0]
    if modeLand in table.keys():
        prob = table[modeLand]
    else:
        prob = 0.05
    return prob


#Computes a movement probability (probability that a relative vector occurs in the sequence of a track)
def moveProb(v, V):
    c = 0
    #Count the number of occurrences of this movement in the track
    for vi in V:
        if vi.equals(v):
        #if (vi.x == v.x) & (vi.y == v.y):
            c+=1
    return float(c)/float(len(V))


def locProb(v, end, land, table, lag):
    newLoc = Vplus(end,v)
    p = getLUProb(newLoc, land, table, lag)
##    for vi in V:
##        if vi.equals(v):
##            p = float(locDistr['overallProb'][V.index(vi)])
    return  p


# Computes a combined probability out of movement and location probablity
def probability(v, V, end, land, table, lag=30):
    return moveProb(v, V)*locProb(v, end, land, table, lag)


# Generates a location probability by using raster land use base map
def locTable(track, land, lag):
    #convert point into line: line is equivalent to infinite number of sampling point to query land use type
    trackLine = geometry.LineString(track)
    #create line buffer
    trackBuffer = trackLine.buffer(lag)
    categoryCount = zonal_stats(trackBuffer, land, categorical=True)  # count of each class pixels
    stats = zonal_stats(trackBuffer, land)  # stats of total count
    total = stats[0]['count']  # total
    
    categoryProb = {k: v / total for k, v in categoryCount[0].items()}  # Normailze to probability
    # Most frequent land use type as the mode    
    return categoryProb
    

##def locProb(v,end):
##    return 0.5
##    BBG2012_Publicatiebestand = 'data\\BBG2012_Publicatiebestand.tif'
##    NDV, xsize, ysize, GeoT, Projection, DataType = gr.get_geo_info(BBG2012_Publicatiebestand)
##    data = gr.load_tiff(BBG2012_Publicatiebestand)
##    s = data.shape
##    rows = s[0]
##    cols = s[1]
##    #load raster data once
##    row, col = lookupWGS84(x, y, GeoT)
##    if row != 'NN' and (row < rows and col < cols):
##                    pass
##    else:
##                    print("Coordinates out of bounds!")


#Computes movement similarity based on chi square contingency table of movement probabilities of relative vectors in two tracks
def moveSim(track, test):
    Vtrack, vsettrack = getV(track)
    #print [str(v) for v in Vtrack]
    Vtest, vsettest = getV(test)
    #print  [str(v) for v in Vtest]

    contingencytable = []
    for v in vsettest:
        #print (str(v)+""+str(moveProb(v, Vtest)) +""+ str(moveProb(v, Vtrack)))
        contingencytable.append([moveProb(v, Vtest) , moveProb(v, Vtrack)])
        #psum += math.pow((moveProb(v, Vtrack) - p2),2)/p2
    #chi = psum * track.size
    print(np.array(contingencytable))
    chi2_stat, p_val, dof, ex = stats.chi2_contingency(np.array(contingencytable))
    print("p_val:"+str(p_val))
    return p_val


def locSim(track, test, table, land, lag):
    Vtrack, vsettrack = getV(track)
    Vtest, vsettest = getV(test)

    contingencytable = []
    testTable = locTable(test, land, lag)
    contingencytable = [np.array((testTable[key], 
                                  table[key])) for key, val in table.items()]
    print(np.array(contingencytable))
    chi2_stat, p_val, dof, ex = stats.chi2_contingency(np.array(contingencytable))
    print("p_val:"+str(p_val))
    return p_val


# Computes a track similarity based on movement similarity and location similarity
def similarity(track, test, table, land, lag):
    return moveSim(track, test)+locSim(track, test, table, land, lag)


#Extends a track (on one end) based on a probability distribution over relative vectors (movements) in the track sequence
def ExtendMimic(track, track0, land, p, lag):
    #Choose an end of the track  (right now only the last point)
    endindex = track.size-1#int(round(float(randint(0,track.size))/float(track.size))*(track.size-1))
    end = track[endindex]
    V, Vset = getV(track)
    faketrack = track
    table = locTable(track0, land, lag)
    #Generate 1 .. max(0.7*track.size) new fake points
    for i in range(0,randint(1,int(0.5*track.size))):
        #q = Q.PriorityQueue()
        #for v in Vset:
            #print "prob:"+str(probability(v,end, V))
            #q.put((probability(v,end, V)-1,Vplus(end,v)))

        probdist = [probability(v, V, end, land, table, lag) for v in Vset]
        if sum(probdist)==1:
            probdist = [i/sum(probdist) for i in probdist]  # Normalize to 1
        else:
            probdist = [1/len(probdist) for i in probdist]
        #print probdist
        #print Vset
        #while not q.empty():
        print(len(Vset))
        for i in range(1,len(Vset)):
            #candidate = q.get()[1]
            #This generates a new point candidate based on random choice of relative vectors over movement probability
            candidate =  Vplus(end,Vset[np.random.choice(np.arange(len(Vset)), None, p=list(probdist))])
            print(candidate)
            test = faketrack.copy()
            test.loc[endindex+1] = candidate  # adding a row to the end of the dataframe  (cumbersome because of geopandas)
            test = test.reset_index(drop=True)  # sorting by index
            if (candidate not in faketrack) :#& (similarity(track, test) > p):
                print("Extend track with:"+str(candidate))
                faketrack = test
                end = candidate
                endindex =test.size-1
                error = False
                break
            else:
                error = True
        if error:
            print("Error: no sufficiently similar candidate found!")
    return faketrack


"""Functions for template masking"""

#Generates a von Neuman neighborhood template (a set of relative point vectors) with radius r
def vNTemplate(r):
    template = []
    for x in range(-r,r+1):
        for y in range(-r,r+1):
            if abs(x) + abs(y) <= r:
                template.append(Point(x,y))
    print([str(p) for p in template])
    return template

#Randomly shifts the centerpoint of the template. Prefers points at the periphery of the template
def rdmshift(template):
    #This random selection prefers center points sitting at the periphery of the template
    templatedistances = [float(max(abs(p.x),abs(p.y))+1) for p in template]
    templateprobabilities = [d/sum(templatedistances) for d in templatedistances]
    v0 = template[np.random.choice(np.arange(len(template)), None, p=templateprobabilities)]
    template = [Vminus(v, v0) for v in template]
    #print [str(p) for p in template]
    return  template

#Applies a template to a geographic point using the rounding increment
def apply(vgeo, template,m):
    template = [Vplus(vgeo, Vmult(m,v)) for v in template]
    #print [str(p) for p in template]
    return  template

#Masks each point in a track using some template
def Masking(track, m, r, k, d):
    out = []
    Pred = []
    vntemplate = vNTemplate(r)
    for v in track:
        print("Now masking point: "+str(v) )
        #Randomly shift template
        temp = apply(v, rdmshift(vntemplate),m)
        #print temp
        candidates = [vk for vk in temp if vk not in Pred]

        #Mask
        candidates.append(v)
        print([str(p) for p in candidates])
        mask = candidates


        out.extend(mask)
        Pred = mask

    print(str(len(out))+" masked points from originally "+str(track.size))
    outgdf = pointlist2GDF(out)
    return outgdf

#Turns point list into geopandas data frame
def pointlist2GDF(pointlist):
        rt = pd.DataFrame({'points': pointlist})
        outputframe = gpd.GeoDataFrame(rt, geometry='points')['points']
        return outputframe


"""Main crowding function"""
def Crowd(track, land, numHome, k=10, p = 0.02, lag=50) :
    track['points'] = list(zip(track.X, track.Y))
    track['points']  = track['points'].apply(Point)
    trackgdf = gpd.GeoDataFrame(track, geometry='points')
    trackgdf.crs = {"init": 'epsg:4326'}
    trackgdf = trackgdf.to_crs({"init": 'epsg:28992'})
    trackgdf.to_file(driver = 'ESRI Shapefile', filename = 'track.shp')
    #print trackgdf

    #Maximum distance parameter
    distances = getDistances(trackgdf['points'])
    d = 1.3* max(distances)
    print('the distance parameter is '+str(d))

    #Rounding increment
    m = math.floor(d/(math.sqrt(2 * k/math.pi)))
    print('the rounding increment is '+str(m))

    #Template radius
    r = int(math.ceil((-1 + math.sqrt(1+4*k))/2))
    print('the template radius is '+str(r))
    
    #Get real home location
    pArray = np.array([[pt.x, pt.y] for ind, pt in np.ndenumerate(trackgdf['points'].values)])
    if not numHome:
        home = np.zeros((2,))
    else:
        home = pArray[-numHome:].mean(axis=0)

    #Start of the programming logic
    lookup,rastertrack = Rasterize(trackgdf['points'],m)
    rastertrack.to_file(driver = 'ESRI Shapefile', filename = 'rastertrack.shp')


    faketrack = ExtendMimic(rastertrack,trackgdf['points'],land,p,lag)
    print(faketrack)
    faketrack.to_file(driver = 'ESRI Shapefile', filename = 'faketrack.shp')

    maskedtrack = Masking(faketrack,m, r, k, d)
    maskedtrack.to_file(driver = 'ESRI Shapefile', filename = 'maskedtrack.shp')
    
    return home












def run(f):
##    track='data\\2420.csv'
##    df = pd.read_csv(track)
##    for track, trackdf in df.groupby("track"):
##        trackdf.to_csv('data\\'+str(track)+".csv")
    
#    df = pd.read_csv('data\\332541.csv')
#    layer = 'results\\sampleLandUse.shp'
#    land = gpd.GeoDataFrame.from_file(layer)
#    track = df[['X','Y']]
#    Crowd(track, land, layer)
    
#    repo_dir = Path('__file__').parents[0]
#    data_dir = repo_dir / 'data'
    
#    layer = 'sampleLandUse.shp'
#    land = gpd.GeoDataFrame.from_file(layer)
    land = 'landUse.tif'
    NDV, xsize, ysize, GeoT, Projection, DataType = gr.get_geo_info(land)
    
    df = pd.read_csv(f)
    numHome = 0
    for index, row in df.iterrows():
        if row['purpose'] == 'home': #or row['purto']=='home':
            numHome += 1
    
    track = df[['X','Y']]
    
    start = time.clock()
    home = Crowd(track, land, numHome, k=10, p = 0.02, lag=50)
    speed = time.clock()-start
    
    return home, speed




#if __name__ == '__main__':
#    main()
