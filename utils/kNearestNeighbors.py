# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Basic libraries
#
import numpy  as np
import pandas as pd

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Scipy library
#
from   scipy.spatial.distance import cdist
from   scipy.spatial.distance import squareform

class KNearestNeighbors():
    """
        @params
            k: int
                Determines k nearest neighbors
            metric: str
                distance metric (KL, JS, Euclidean)
    """
    def __init__(self, k = [10], metrics = ['euclidean'], verbose = False):
        """
            @params
                x: np.ndarray
                    An m x n array with m samples and n dimensions
        """        
        self._k       = k
        self._metrics = metrics
        self._verbose = verbose
        
        if (self._verbose): 
            print('[INFO] Process setup')
            print('***************************')
            print('[INFO] k       = ', self._k)
            print('[INFO] Metrics = ', self._metrics)

            
    def calculateDistances(self, data):
        """
            Returns the mean pairwise distance between the k'th nearest neighbors
            @params
                x: np.ndarray
                    An m x n array with m samples and n dimensions
        """
        self._data      = data
        
        # Set dictionary with distances
        self._distances = dict()
        
        for metric in self._metrics:
            self._distances[ metric ] = cdist(data, self._X, metric)
            if (self._verbose): print('[INFO] Metric: ', metric)
        
        if (self._verbose): print('[INFO] Distance matrix contructed')
            

    
    def __call__(self, X, T, Y):
        self._X = X
        self._Y = Y
        #
        self._Treatment = T

        
        if (self._verbose): print('[INFO] Database establisted')
             
    
    def getEstimatedOutcomes(self):
        
        nearestInstances = [] 
        
        
        for i in range( self._data.shape[0] ):
        
            Y0, Y1 = [], []
            for metric in self._metrics:
                df = pd.DataFrame({})
                # Get info: Treatment/Control
                df['Treatment'] = self._Treatment            
                # Get info: Y
                df['Y']         = self._Y
                # Get distances
                df['Distances'] = self._distances[ metric ][i,:]
            
                # Sort array by distances
                df.sort_values(by = 'Distances', inplace = True)

                # Remove the distance of the instance with itself
                df = df[df['Distances'] > 0]
            
                for k in self._k:
                    # Instance belongs to Control group
                    Y0 += [ df[df['Treatment'] == 0]['Y'][:k].mean() ]
                    # Instance belongs to Treatment group
                    Y1 += [ df[df['Treatment'] == 1]['Y'][:k].mean() ]


            nearestInstances += [Y0 + Y1]

        # Convert to ndarray
        nearestInstances = np.asarray( nearestInstances )
        if (self._verbose): print('[INFO] Array with nearest neighbors instances contructed\n')
        
        return (nearestInstances)
    

    def getEstimatedOutcomesAndDistances(self):
        
        nearestInstances = [] 
        for i in range( self._data.shape[0] ):
            
            df = pd.DataFrame({})
            # Get info: Treatment/Control
            df['Treatment'] = self._Treatment            
            # Get info: Y
            df['Y']         = self._Y
            # Get distances
            df['Distances'] = self._distances[i,:]
            # Sort array by distances
            df.sort_values(by = 'Distances', inplace = True)

            # Remove the distance of the instance with itself
            df = df[df['Distances'] > 0]
            
            # Instance belongs to Control group
            Y0 = df[df['Treatment'] == 0]['Y'][:self._k].mean()
            D0 = df[df['Treatment'] == 0]['Distances'][:self._k].mean()
            # Instance belongs to Treatment group
            Y1 = df[df['Treatment'] == 1]['Y'][:self._k].mean()
            D1 = df[df['Treatment'] == 1]['Distances'][:self._k].mean()
            
            nearestInstances += [ [Y0, D0,
                                   Y1, D1] ]

        # Convert to ndarray
        nearestInstances = np.asarray( nearestInstances )
        if (self._verbose): print('[INFO] Array with nearest neighbors instances contructed')
        
        return (nearestInstances)
    
    
    def getEstimatedOutcomesStatistics(self):
        
        nearestInstances = [] 
        for i in range( self._data.shape[0] ):
            
            df = pd.DataFrame({})
            # Get info: Treatment/Control
            df['Treatment'] = self._Treatment            
            # Get info: Y
            df['Y']         = self._Y
            # Get distances
            df['Distances'] = self._distances[i,:]
            # Sort array by distances
            df.sort_values(by = 'Distances', inplace = True)

            # Remove the distance of the instance with itself
            df = df[df['Distances'] > 0]
            
            # Instance belongs to Control group
            #
            Y0_mean   = df[df['Treatment'] == 0]['Y'][:self._k].mean()
            Y0_std    = df[df['Treatment'] == 0]['Y'][:self._k].std()
            Y0_median = df[df['Treatment'] == 0]['Y'][:self._k].median()
            Y0_min    = df[df['Treatment'] == 0]['Y'][:self._k].min()
            Y0_max    = df[df['Treatment'] == 0]['Y'][:self._k].max()
#             Y0_skew  = df[df['Treatment'] == 0]['Y'][:self._k].skew()
#             Y0_kurt  = df[df['Treatment'] == 0]['Y'][:self._k].kurt()
            
            # Instance belongs to Treatment group
            Y1_mean   = df[df['Treatment'] == 1]['Y'][:self._k].mean()
            Y1_std    = df[df['Treatment'] == 1]['Y'][:self._k].std()
            Y1_median = df[df['Treatment'] == 1]['Y'][:self._k].median()            
            Y1_min    = df[df['Treatment'] == 1]['Y'][:self._k].min()
            Y1_max    = df[df['Treatment'] == 1]['Y'][:self._k].max()
#             Y1_skew  = df[df['Treatment'] == 1]['Y'][:self._k].skew()
#             Y1_kurt  = df[df['Treatment'] == 1]['Y'][:self._k].kurt()


            nearestInstances += [ [Y0_mean, Y0_std, Y0_median, Y0_min, Y0_max, \
                                   Y1_mean, Y1_std, Y1_median, Y1_min, Y1_max, ] ]

        # Convert to ndarray
        nearestInstances = np.asarray( nearestInstances )
        if (self._verbose): print('[INFO] Array with nearest neighbors instances contructed')
        
        return (nearestInstances)    
    
    
    
    def getEstimatedWeightedOutcomes(self):
        
        nearestInstances = [] 
        for i in range( self._data.shape[0] ):
            
            df = pd.DataFrame({})
            # Get info: Treatment/Control
            df['Treatment'] = self._Treatment            
            # Get info: Y
            df['Y']         = self._Y
            # Get distances
            df['Distances'] = self._distances[i,:]
            # Sort array by distances
            df.sort_values(by = 'Distances', inplace = True)

            # Remove the distance of the instance with itself
            df = df[df['Distances'] > 0]
            
            # Instance belongs to Control group
            #
            temp_df = df[df['Treatment'] == 0][:self._k].copy()
            #
            # Calculate Wi
            temp_df['W'] = 1.0 / temp_df['Distances'] 
            temp_df['W'] = temp_df['W'] / temp_df['W'].sum()
            Y0 = (temp_df['Y'] * temp_df['W']).sum()

            # Instance belongs to Treatment group
            #
            temp_df = df[df['Treatment'] == 1][:self._k].copy()
            #
            # Calculate Wi
            temp_df['W'] = 1.0 / temp_df['Distances'] 
            temp_df['W'] = temp_df['W'] / temp_df['W'].sum()
            Y1 = (temp_df['Y'] * temp_df['W']).sum()
            
            nearestInstances += [ [Y0, Y1] ]

        # Convert to ndarray
        nearestInstances = np.asarray( nearestInstances )
        if (self._verbose): print('[INFO] Array with nearest neighbors instances contructed')
        
        return (nearestInstances)
    
    
    
    
    
    