#-------------------------------------------------------------------------------
# Name:        Data selection and screening
#                             
# Purpose:     Functions for selecting tracks.
#
# Author:      Jiong (Jon) Wang
#
# Created:     20/09/2018
# Copyright:   
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
import pandas as pd

from pathlib import Path
from scipy import stats
from scipy.spatial import Voronoi

# Extract home number=num tracks from or to home from one user
def homeTrack(df, num):
    track_ind = []
    for index, row in df.iterrows():
        if row['purpose'] == 'home': #or row['purto']=='home':
            track_ind.append(row['track'])
            
    hometrack_ind = np.random.choice(track_ind, num)
    print(len(hometrack_ind))
    for ind in hometrack_ind:
        df.loc[df['track'] == ind].to_csv(str(ind)+'.csv', header=True)
    return None


# Loop over all user tracks in destination folder to extrack home tracks
def extractTrack(data_dir, num):
    for f in Path(data_dir).glob('*.csv'):
        try:
            df = pd.read_csv(f)
            print(f)
            if df.shape[0] > 300:
                homeTrack(df, num)
        except:
            continue

    return 
    

def main():
    
    repo_dir = Path('__file__').parents[0]
    data_dir = repo_dir / 'data'
    
    num = 4  # Number of home tracks from each user
    
    extractTrack(data_dir, num)
    
    

if __name__ == '__main__':
    main()