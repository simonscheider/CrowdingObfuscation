#-------------------------------------------------------------------------------
# Name:        Evaluating robustness of obfucated tracks
# Purpose:     Functions for comparing attacked/predicted tracks with the original track.
#              Main method is Evaluate().
#              Metrics include:
#              1. line pattern based upon track buffer
#              2. endpoint locatoin based upon ellipse
#              3. endpoint pattern based upon distribution centroid
#
# Author:      Jiong Wang
#
# Created:     29/10/2018
# Copyright:   (c) Schei008 2018
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import geopandas as gpd
import math
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from shapely import geometry
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from random import randint
from scipy import stats

# The Jaccard Index as combined metric of precision and recall
def jaccard(poly, polypred):
    if poly.intersects(polypred):
        intersection = poly.intersection(polypred).area
        union = poly.area+polypred.area-poly.intersection(polypred).area
        jaccard = intersection/union
    return jaccard


def fractaldim(pointlist, boxlevel):
        """Returns the approximate fractal dimension of pointlist,
via the box-counting algorithm.  The elements of pointset
should be three-element sequences of numbers between 0.0
and 1.0.  The boxlevel is the number of divisions made on
each dimension, and should be greater than 1."""

        if boxlevel <= 1.0: return -1

        pointdict = {}

        def mapfunction(val, level=boxlevel):
                return int(val * level)

        for point in pointlist:
                box = (int(point[0] * boxlevel),
                        int(point[1] * boxlevel),
                        int(point[2] * boxlevel))
                #box = tuple(map(mapfunction, point))
                if 'box'not in pointdict.keys():
                        pointdict[box] = 1

        num = len(pointdict.keys())

        if num == 0: return -1

        return math.log(num) / math.log(boxlevel)


# Track precision using the Jaccard Index (as combined measurement of precsion and recall)
def linePattern(track, trackpred, lag):
    #Create line from track points
    trackLine = geometry.LineString(track)
    trackpredLine = geometry.LineString(trackpred)
    #create line buffer
    trackBuffer = Polygon(trackLine.buffer(lag))
    trackpredBuffer = Polygon(trackpredLine.buffer(lag))
    return jaccard(trackBuffer, trackpredBuffer)


#def drawEllip(track, shape=1.02):
#    #Extract endpoint values as the focii of ellipse
#    a1 = track[0].x
#    b1 = track[0].y
#    a2 = track[len(track)-1].x
#    b2 = track[len(track)-1].y
#    c = math.sqrt((a2-a1)**2+(b2-b1)**2)*shape
#    
#    # Compute ellipse parameters
#    a = c / 2                                # Semimajor axis
#    x0 = (a1 + a2) / 2                       # Center x-value
#    y0 = (b1 + b2) / 2                       # Center y-value
#    f = np.sqrt((a1 - x0)**2 + (b1 - y0)**2) # Distance from center to focus
#    b = np.sqrt(a**2 - f**2)                 # Semiminor axis
#    phi = np.arctan2((b2 - b1), (a2 - a1))   # Angle betw major axis and x-axis
#    
#    # Parametric plot in t
#    resolution = 1000
#    t = np.linspace(0, 2*np.pi, resolution)
#    x = x0 + a * np.cos(t) * np.cos(phi) - b * np.sin(t) * np.sin(phi)
#    y = y0 + a * np.cos(t) * np.sin(phi) + b * np.sin(t) * np.cos(phi)
#    ellipse = Polygon(list(zip(x, y)))
#    return ellipse
#    
#
#def endLoc(track, trackpred):
#    #Create ellipsoid by treating end points as focii
#    trackellip = drawEllip(track)
#    trackpredellip = drawEllip(trackpred)
#    return jaccard(trackellip, trackpredellip)

# Search for k neighbors of end points for track end pattern analysis
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


def endPattern(track, home, k=5):
    track = np.array([[pt.x, pt.y] for ind, pt in np.ndenumerate(track.values)])
    distances, indices = knn(track, k)
    
    #start centroid and end centroid according to distribution of track points
    #trackst_c = centroid(track[indices[0]].T)
    trackend_c = centroid(track[indices[len(track)-1]].T)     
    return np.linalg.norm(trackend_c-home)


def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]


# Draw SDE around p end points of a track 
def drawEllipse(track, p):
    track = np.array([[pt.x, pt.y] for ind, pt in np.ndenumerate(track.values)])
    
    x = track[-p:,0].T
    y = track[-p:,1].T
    
    nstd = 2
    ax = plt.subplot(111)
    
    cov = np.cov(x, y)
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w, h = 2 * nstd * np.sqrt(vals)
    ell = Ellipse(xy=(np.mean(x), np.mean(y)),
                  width=w, height=h,
                  angle=theta, color='black')
    #ell.set_facecolor('none')
    ax.add_artist(ell)
    #plt.scatter(x, y)
    #plt.show()
    return w, h, theta

# Percentage of difference in spread (in two axies of the ellipse) and direction
def ellipsePattern(track, trackpred, p):
    diffx, diffy, difftheta = np.divide(np.subtract(drawEllipse(trackpred,p), drawEllipse(track,p)), drawEllipse(track,p))
    return abs(diffx), abs(diffy), abs(difftheta)

def ellipseFoci(track, trackpred):
    track = np.array([[pt.x, pt.y] for ind, pt in np.ndenumerate(track.values)])
    trackpred = np.array([[pt.x, pt.y] for ind, pt in np.ndenumerate(trackpred.values)])

    # Draw ellipse of the original track
    x = track[:,0].T
    y = track[:,1].T
    nstd = 2    
    cov = np.cov(x, y)
    
    vals, vecs = eigsorted(cov)
    w, h = 2 * nstd * np.sqrt(vals)
    unitSemiMajor = (vecs[:,0][::-1])/math.sqrt(vecs[:,0][::-1][0]**2+vecs[:,0][::-1][0]**2)

    # Find forcii as start and end (home location)
    predictstart = math.sqrt(w**2-h**2)*unitSemiMajor + np.array((np.mean(x), np.mean(y)))
    predictend = np.array((np.mean(x), np.mean(y))) - math.sqrt(w**2-h**2)*unitSemiMajor
    
    start = trackpred[0][0], trackpred[0][1]
    end = trackpred[len(trackpred)-1][0], trackpred[len(trackpred)-1][1]
    
    # Evaluate the difference
    return vertexDiff(start, predictstart), vertexDiff(end, predictend)
    

def vertexDiff(point1, point2):
    dist = np.hypot(point1, point2)
    return dist


def Evaluate(track, trackFake, home, lag=50, p=5):
    
    #Complexity of the original track
    t = np.array([[pt.x, pt.y] for ind, pt in np.ndenumerate(track.values)])
    t = list([tuple(row)+(0,) for row in t])
    frac = fractaldim(t, 8)
    
    #Similarity between original and attacked/predicted track
    p0 = linePattern(track, trackFake, lag)
    print('the attacking precision on LINE PATTERN is '+str(p0))
    
    #Similarity between original and attacked/predicted END clusters
    p2_1 = endPattern(trackFake, home, k=5)
    print('the attacked/predicted home is '+str(p2_1)+' meters from the original one')
    
    #Similarity between original and attacked/predicted Standard Deviation Ellipse
    p3_1, p3_2, p3_3 = ellipsePattern(track, trackFake, p)
    print('the SPREADING in semimajor, semiminor and DIRECTION of attacked/predicted track is '+str(p3_1)+', '+str(p3_2)+', '+str(p3_3)+' percent different from the original track.')
#    p4_1, p4_2 = ellipseFoci(track, trackpred)
#    print('the ends of attacked/predicted track is '+str(p4_1)+', and'+str(p4_2)+' meters different from the original track.')

#    #Similarity between original and attacked/predicted endpoints location
#    p1 = endLoc(track, trackpred)
#    print('the attacking precision on ENDPOINT LOCATION is '+str(p1))
    
    return np.array([len(track), frac, p0, p2_1, p3_1, p3_2, p3_3])



def evalPreserve(originaltrack, faketrack, home):
#    originaltrack = 'track.shp'
#    predictedtrack = 'attacked.shp'
#    
    track = gpd.GeoDataFrame.from_file(originaltrack)
    trackFake = gpd.GeoDataFrame.from_file(faketrack)
    summary = Evaluate(track['geometry'], trackFake['geometry'], home, lag=50, p=5)
    return summary




if __name__ == '__main__':
    main()
