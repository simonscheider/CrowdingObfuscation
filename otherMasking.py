#-------------------------------------------------------------------------------
# Name:        Basic Masking: Random Perturbation
#                             Voronoi Masking
# Purpose:     Functions for masking spatial point cloud.
#
# Author:      Jiong (Jon) Wang
#
# Created:     20/09/2018
# Copyright:   
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas import GeoSeries
import fiona
from fiona.crs import from_epsg
from sklearn.neighbors import NearestNeighbors
from shapely.geometry import Point, LineString, mapping
import shapely
from scipy import stats
from scipy.spatial import Voronoi
from pathlib import Path
import matplotlib.pyplot as plt


# Find K nearest neighbors at each of the track points
def knn(pArray, k):
    #Table of k-neighbors of each track points
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(pArray)
    distances, indices = nbrs.kneighbors(pArray)
    return distances, indices


# Relocate each point based upon kernel density estimation of its K-neighborhood
def relocate(cluster):
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
    relocated = positions.T[np.random.choice(np.arange(len(densityCol)), None, p=list(densityCol))]
    return relocated


# Visualize density distribution of K neighbors cluster
def plotDensity(xmin, xmax, ymin, ymax, cluster, density, positions, X):
    Z = np.reshape(density(positions).T, X.shape)
    fig, ax = plt.subplots()
    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
    ax.plot(cluster[0], cluster[1], 'r.', markersize=5)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    plt.show()
    
    
# Gaussian perturbation
def GaussianPb(track, k):
    pArray = np.array([[pt.x, pt.y] for ind, pt in np.ndenumerate(track.values)])
    distances, indices = knn(pArray, k)
    #Weighted track perturbation based upon Gaussian as weights
    pbTrack = []
    for ind in indices:
        #Kernel density for each K neighborhood cluster
        cluster = pArray[ind].T
        relocated = relocate(cluster)
        pbTrack.append(relocated.T)
    pbTrack = np.array(pbTrack)
    pbTrack = GeoSeries(map(Point, pbTrack))
    return pbTrack


def vorSnap(lines, p):
    dist = p.distance(lines[0])
    global closestP
    for j in range(len(lines)-1):
        line = lines[j]
        if p.distance(line)<dist:
            closestP = line.interpolate(line.project(p))
            dist = p.distance(line)  
    return np.array(closestP)
    

def VorMask(track):
    pArray = np.array([[pt.x, pt.y] for ind, pt in np.ndenumerate(track.values)])
    vor = Voronoi(pArray)
    lines = [LineString(vor.vertices[line]) for line in vor.ridge_vertices if -1 not in line]
    
    #Save the Voronoi polygons in shp
    voronoi = [poly for poly in shapely.ops.polygonize(lines)]
    schemaVor = {'geometry': 'Polygon','properties': {'Id':'int:5'},}
    with fiona.open("vor.shp", "w", driver='ESRI Shapefile', crs=from_epsg(28992), schema=schemaVor) as output:
        for v in voronoi:
            output.write({'geometry': mapping(v), 'properties': {'Id':voronoi.index(v)}})
    
    vorTrack = []
    for i in range(len(pArray)):
        p = pArray[i]
        p = Point(p)
        vorTrack.append(vorSnap(lines, p))          
    vorTrack = GeoSeries(map(Point, np.array(vorTrack)))
    return vorTrack
        

def run(f):

    df = pd.read_csv(f)
    track = df[['X','Y']]    

    track['points'] = list(zip(track.X, track.Y))
    track['points']  = track['points'].apply(Point)
    trackgdf = gpd.GeoDataFrame(track, geometry='points')
    trackgdf.crs = {"init": 'epsg:4326'}
    trackgdf = trackgdf.to_crs({"init": 'epsg:28992'})
    # Alternative: filename = str(f).split('.')[0][10:]+'orig.shp'
    trackgdf.to_file(driver = 'ESRI Shapefile', filename = 'orig.shp')
    
    k = 6
    pbTrack = GaussianPb(trackgdf['points'], k)
    pbTrack.to_file(driver = 'ESRI Shapefile', filename = 'Gaussian.shp')
    
    vorTrack = VorMask(trackgdf['points'])
    vorTrack.to_file(driver = 'ESRI Shapefile', filename = 'Vor.shp')

    

#if __name__ == '__main__':
#    main()
