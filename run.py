#-------------------------------------------------------------------------------
# Name:        Data selection and screening
#                             
# Purpose:     Functions for selecting tracks.
#
# Author:      Jiong (Jon) Wang
#
# Created:     16/01/2019
# Copyright:   
# Licence:     <your licence>
#-------------------------------------------------------------------------------


import dataSelection
dataSelection.main()

from pathlib import Path
import numpy as np
import otherMasking
import crowdingRaster
import attackRaster
import evaluationReconstruct
import evaluationExposure
import os
import pandas as pd

repo_dir = Path('__file__').parents[0]
data_dir = repo_dir / 'Extracted'

i = 1
reconstructC=[]
preserveC=[]
reconstructG=[]
preserveG=[]
reconstructV=[]
preserveV=[]
for f in Path(data_dir).glob('*.csv'):
    print('working on track '+ str(f) + ' and is the number ' + str(i) + ' out of the total !')
    try: 
        otherMasking.run(f)
        home, speed = crowdingRaster.run(f)
        
        #Crowding on original track
        potentialHome = attackRaster.attacking(fake='faketrack.shp')
        #Reconstruction
        originaltrack = 'track.shp'
        predictedtrack = 'attacked.shp'
        reconSummary = evaluationReconstruct.evalRecon(originaltrack, predictedtrack, home, potentialHome)
        reconSummary = np.append(reconSummary, speed/reconSummary[0])
        reconstructC.append(reconSummary.T)
        #Preservation
        originaltrack = 'track.shp'
        faketrack = 'faketrack.shp'
        concentration = 'no2small.tif'
        preserveSummary = evaluationExposure.evalPres(originaltrack, faketrack, concentration)
        preserveC.append(preserveSummary.T)

        
        #Other masking NO.1
        potentialHome = attackRaster.attacking(fake='Gaussian.shp')
        #Reconstruction
        originaltrack = 'track.shp'
        predictedtrack = 'attacked.shp'
        reconSummary = evaluationReconstruct.evalRecon(originaltrack, predictedtrack, home, potentialHome)
        reconSummary = np.append(reconSummary, speed/reconSummary[0])
        reconstructG.append(reconSummary.T)
        #Preservation
        originaltrack = 'track.shp'
        faketrack = 'Gaussian.shp'
        concentration = 'no2small.tif'
        preserveSummary = evaluationExposure.evalPres(originaltrack, faketrack, concentration)
        preserveG.append(preserveSummary.T)
        
        #Other masking NO.2
        potentialHome = attackRaster.attacking(fake='Vor.shp')
        #Reconstruction
        originaltrack = 'track.shp'
        predictedtrack = 'attacked.shp'
        reconSummary = evaluationReconstruct.evalRecon(originaltrack, predictedtrack, home, potentialHome)
        reconSummary = np.append(reconSummary, speed/reconSummary[0])
        reconstructV.append(reconSummary.T)
        #Preservation
        originaltrack = 'track.shp'
        faketrack = 'Vor.shp'
        concentration = 'no2small.tif'
        preserveSummary = evaluationExposure.evalPres(originaltrack, faketrack, concentration)
        preserveV.append(preserveSummary.T)
        
        os.remove('track.shp')
        os.remove('faketrack.shp')
        os.remove('attacked.shp')
        os.remove('Gaussian.shp')
        os.remove('Vor.shp')
        
        i+=1
    except:
        continue

df_preC = pd.DataFrame(columns=['min', 'max', 'median', 'mean'], data = preserveC)
df_reconC = pd.DataFrame(columns=['len', 'frac', 'lineSim', 'homeDis', 'majorPtgDiff', 'minorPtgDiff', 'thetaPtgDiff', 'time'], data = reconstructC)
df_preC.to_csv('preserveC.csv', sep='\t')
df_reconC.to_csv('reconstructC.csv', sep='\t')

df_preG = pd.DataFrame(columns=['min', 'max', 'median', 'mean'], data = preserveG)
df_reconG = pd.DataFrame(columns=['len', 'frac', 'lineSim', 'homeDis', 'majorPtgDiff', 'minorPtgDiff', 'thetaPtgDiff', 'time'], data = reconstructG)
df_preG.to_csv('preserveG.csv', sep='\t')
df_reconG.to_csv('reconstructG.csv', sep='\t')

df_preV = pd.DataFrame(columns=['min', 'max', 'median', 'mean'], data = preserveV)
df_reconV = pd.DataFrame(columns=['len', 'frac', 'lineSim', 'homeDis', 'majorPtgDiff', 'minorPtgDiff', 'thetaPtgDiff', 'time'], data = reconstructV)
df_preV.to_csv('preserveV.csv', sep='\t')
df_reconV.to_csv('reconstructV.csv', sep='\t')


##################################
# Plotting
##################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

reconC = pd.read_csv('reconstructC.csv', sep='\s+')
reconG = pd.read_csv('reconstructG.csv', sep='\s+')
reconV = pd.read_csv('reconstructV.csv', sep='\s+')

preC = pd.read_csv('preserveC.csv', sep='\s+')
preG = pd.read_csv('preserveG.csv', sep='\s+')
preV = pd.read_csv('preserveV.csv', sep='\s+')

# Line characteristics distribution
plt.subplot(1,2,1)
x = reconC['len'].values
sns.distplot(x, bins=30)
plt.xlabel('Track length (track points)')
plt.ylabel('Density')
plt.subplot(1,2,2)
x = reconC['frac'].values
sns.distplot(x, bins=30)
plt.xlabel('Track complexity (fractal dimensions)')
plt.ylabel('Density')
plt.savefig('Fig01.png', dpi=300)


# Reconstruction comparison for all obfuscation techniques
x = reconC['lineSim'].values
y = reconG['lineSim'].values
z = reconV['lineSim'].values
sns.kdeplot(x, shade=True, color="lightcoral", label="Attack accuracy of crowded track")
sns.kdeplot(y, shade=True, color="yellowgreen", label="Attack accuracy of perturbed track")
sns.kdeplot(z, shade=True, color="lightskyblue", label="Attack accuracy of masked track")
plt.savefig('Fig02.png', dpi=300)


# Home location attacking comparison for all obfuscation techniques
x = reconC['homeDis'].values
y = reconG['homeDis'].values
y = y[y<1200]
z = reconV['homeDis'].values
z = z[z<1500]
sns.kdeplot(x, shade=True, color="lightcoral", label="Home attack accuracy of crowded track (m)")
sns.kdeplot(y, shade=True, color="yellowgreen", label="Home attack accuracy of perturbed track (m)")
sns.kdeplot(z, shade=True, color="lightskyblue", label="Home attack accuracy of masked track (m)")
plt.savefig('Fig03.png', dpi=300)


# SDE comparison for all obfuscation 
plt.subplot(1,3,1)
x = reconC['majorPtgDiff'].values
y = reconG['majorPtgDiff'].values
z = reconV['majorPtgDiff'].values
sns.kdeplot(x, shade=True, color="lightcoral", label="Attacked crowded loss: Major spread")
sns.kdeplot(y, shade=True, color="yellowgreen", label="Attacked perturbed loss: Major spread")
sns.kdeplot(z, shade=True, color="lightskyblue", label="Attacked masked loss: Major spread")

plt.subplot(1,3,2)
x = reconC['minorPtgDiff'].values
y = reconG['minorPtgDiff'].values
z = reconV['minorPtgDiff'].values
sns.kdeplot(x, shade=True, color="lightcoral", label="Attacked crowded loss: Minor spread")
sns.kdeplot(y, shade=True, color="yellowgreen", label="Attacked perturbed loss: Minor spread")
sns.kdeplot(z, shade=True, color="lightskyblue", label="Attacked masked loss: Minor spread")

plt.subplot(1,3,3)
x = reconC['thetaPtgDiff'].values
y = reconG['thetaPtgDiff'].values
z = reconV['thetaPtgDiff'].values
sns.kdeplot(x, shade=True, color="lightcoral", label="Attacked crowded loss: Direction")
sns.kdeplot(y, shade=True, color="yellowgreen", label="Attacked perturbed loss: Direction")
sns.kdeplot(z, shade=True, color="lightskyblue", label="Attacked masked loss: Direction")
plt.savefig('Fig04.png', dpi=300)


# Preservation of exposure estimation
plt.subplot(1,3,1)
x = preC['mean'].values
y = preG['mean'].values
z = preV['mean'].values
sns.kdeplot(x, shade=True, color="lightcoral", label="Crowded exposure MEAN loss (ptg.)")
sns.kdeplot(y, shade=True, color="yellowgreen", label="Perturbed exposure MEAN loss (ptg.)")
sns.kdeplot(z, shade=True, color="lightskyblue", label="Masked exposure MEAN loss (ptg.)")

plt.subplot(1,3,2)
x = preC['min'].values
y = preG['min'].values
z = preV['min'].values
sns.kdeplot(x, shade=True, color="lightcoral", label="Crowded exposure MIN loss (ptg.)")
sns.kdeplot(y, shade=True, color="yellowgreen", label="Perturbed exposure MIN loss (ptg.)")
sns.kdeplot(z, shade=True, color="lightskyblue", label="Masked exposure MIN loss (ptg.)")

plt.subplot(1,3,3)
x = preC['max'].values
y = preG['max'].values
z = preV['max'].values
sns.kdeplot(x, shade=True, color="lightcoral", label="Crowded exposure MAX loss (ptg.)")
sns.kdeplot(y, shade=True, color="yellowgreen", label="Perturbed exposure MAX loss (ptg.)")
sns.kdeplot(z, shade=True, color="lightskyblue", label="Masked exposure MAX loss (ptg.)")
plt.savefig('Fig05.png', dpi=300)


# Joint evaluation of obfuscation intensity (reconstruction similarity) and exposure accuracy (preservation)
#crowding
x = reconC['lineSim'].values
x = np.delete(x,[1500])
y = preC['mean'].values
g = sns.jointplot(x, y, kind="kde", color="powderblue")
g.plot_joint(plt.scatter, c="lightgray", s=0.5, linewidth=1, marker=".")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("Reconstructed crowded track similarity (Ptg.)", "Accuracy loss due to crowding");
plt.gca().set_xlim(0,1)
plt.gca().set_ylim(-0.05,0.35)
plt.savefig('Fig06_1.png', dpi=300)
#Gaussian perturbation
x = reconG['lineSim'].values
y = preG['mean'].values
g = sns.jointplot(x, y, kind="kde", color="powderblue")
g.plot_joint(plt.scatter, c="lightgray", s=0.5, linewidth=1, marker=".")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("Reconstructed perturbed track similarity (Ptg.)", "Accuracy loss due to perturbation");
plt.gca().set_xlim(0,1)
plt.gca().set_ylim(-0.05,0.35)
plt.savefig('Fig06_2.png', dpi=300)
#Voronoi masking
x = reconV['lineSim'].values
y = preV['mean'].values
g = sns.jointplot(x, y, kind="kde", color="powderblue")
g.plot_joint(plt.scatter, c="lightgray", s=0.5, linewidth=1, marker=".")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("Reconstructed masked track similarity (Ptg.)", "Accuracy loss due to masking");
plt.gca().set_xlim(0,1)
plt.gca().set_ylim(-0.05,0.35)
plt.savefig('Fig06_3.png', dpi=300)


# Joint evaluation of home protection and obfuscation intensity
#crowding
x = reconC['lineSim'].values
y = reconC['homeDis'].values
g = sns.jointplot(x, y, kind="kde", color="burlywood")
g.plot_joint(plt.scatter, c="lightgray", s=0.5, linewidth=1, marker=".")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("Reconstructed crowded track similarity (Ptg.)", "Home protection (distance:m)");
plt.gca().set_xlim(0,1)
plt.gca().set_ylim(0,10000)
plt.savefig('Fig07_1.png', dpi=300)
#Gaussian perturbation
temp = reconG[reconG.homeDis<1200]
x = temp['lineSim'].values
y = temp['homeDis'].values
g = sns.jointplot(x, y, kind="kde", color="burlywood")
g.plot_joint(plt.scatter, c="lightgray", s=0.5, linewidth=1, marker=".")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("Reconstructed perturbed track similarity (Ptg.)", "Home protection (distance:m)");
plt.gca().set_xlim(0,1)
plt.gca().set_ylim(0,10000)
plt.savefig('Fig07_2.png', dpi=300)
#Voronoi masking
temp = reconV[reconV.homeDis<1500]
x = temp['lineSim'].values
y = temp['homeDis'].values
g = sns.jointplot(x, y, kind="kde", color="burlywood")
g.plot_joint(plt.scatter, c="lightgray", s=0.5, linewidth=1, marker=".")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("Reconstructed masked track similarity (Ptg.)", "Home protection (distance:m)");
plt.gca().set_xlim(0,1)
plt.gca().set_ylim(0,10000)
plt.savefig('Fig07_3.png', dpi=300)


# Joint evaluation of obfuscation and track characteristics
#crowding
x = reconC['lineSim'].values
y = reconC['homeDis'].values
g = sns.jointplot(x, y, kind="kde", color="burlywood")
g.plot_joint(plt.scatter, c="lightgray", s=0.5, linewidth=1, marker=".")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("Reconstructed crowded track similarity (Ptg.)", "Home protection (distance:m)");
plt.gca().set_xlim(0,1)
plt.gca().set_ylim(0,10000)
plt.savefig('Fig07_1.png', dpi=300)
#Gaussian perturbation
x = reconG['lineSim'].values
y = reconG['homeDis'].values
g = sns.jointplot(x, y, kind="kde", color="burlywood")
g.plot_joint(plt.scatter, c="lightgray", s=0.5, linewidth=1, marker=".")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("Reconstructed perturbed track similarity (Ptg.)", "Home protection (distance:m)");
plt.gca().set_xlim(0,1)
plt.gca().set_ylim(0,10000)
plt.savefig('Fig07_2.png', dpi=300)
#Voronoi masking
x = reconV['lineSim'].values
y = reconV['homeDis'].values
g = sns.jointplot(x, y, kind="kde", color="burlywood")
g.plot_joint(plt.scatter, c="lightgray", s=0.5, linewidth=1, marker=".")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("Reconstructed masked track similarity (Ptg.)", "Home protection (distance:m)");
plt.gca().set_xlim(0,1)
plt.gca().set_ylim(0,10000)
plt.savefig('Fig07_3.png', dpi=300)


# Joint evaluation of obfuscation and track characteristics
# Crowding
plt.subplot(1,3,1)
x = reconC['len'].values
x = np.delete(x,[1500])
y = reconC['lineSim'].values
y = np.delete(y,[1500])
z = preC['mean'].values
plt.scatter(x, y, s=z*1000, cmap="orange", alpha=0.3, edgecolors="grey", linewidth=1)
plt.xlabel("Track length")
plt.ylabel("Attacking accuracy")
plt.legend('EL')
# Gaussian
plt.subplot(1,3,2)
x = reconG['len'].values
y = reconG['lineSim'].values
z = preG['mean'].values
plt.scatter(x, y, s=z*1000, cmap="orange", alpha=0.3, edgecolors="grey", linewidth=1)
plt.xlabel("Track length")
plt.ylabel("Attacking accuracy")
plt.legend('EL')
# Masking
plt.subplot(1,3,3)
x = reconV['len'].values
y = reconV['lineSim'].values
z = preV['mean'].values
plt.scatter(x, y, s=z*1000, cmap="orange", alpha=0.3, edgecolors="grey", linewidth=1)
plt.xlabel("Track length")
plt.ylabel("Attacking accuracy")
plt.legend('EL')
plt.savefig('Fig08.png', dpi=300)



    
    
    