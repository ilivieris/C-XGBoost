import os
import numpy as np
import pandas as pd
from   sklearn.model_selection import train_test_split


class Synthetic_dataset():
    def __init__(self, path = 'Data/Synthetic/5000/'):
        self.path = path
        self.nProblems = len([file for file in os.listdir(path) if "train" in file])


    def getTraining(self, idx = 0):
        
        # Load Training data
        data = np.load(self.path + 'train{}.npz'.format(idx))

        X = data['x']
        T = data['t']

        Y   = data['yf'].reshape(-1,1)
        Yfc = data['ycf'].reshape(-1,1)

        # mu0 = data['mu0'].reshape(-1,1)
        # mu1 = data['mu1'].reshape(-1,1)

        # potential_Y = np.concatenate( [mu0, mu1], axis=1 )
        potential_Y = np.array( [[x,y] if t == 0 else [y, x] for (t,x,y) in zip(T,Y,Yfc)] ).squeeze()
        
        return (X, T, Y, potential_Y)
    
    
    def getTesting(self, idx = 0):
        
        # Load Testing data
        data = np.load(self.path + 'test{}.npz'.format(idx))

        X = data['x']
        T = data['t']

        Y   = data['yf'].reshape(-1,1)
        Yfc = data['ycf'].reshape(-1,1)

        # mu0 = data['mu0'].reshape(-1,1)
        # mu1 = data['mu1'].reshape(-1,1)

        # potential_Y = np.concatenate( [mu0, mu1], axis=1 )
        potential_Y = np.array( [[x,y] if t == 0 else [y, x] for (t,x,y) in zip(T,Y,Yfc)] ).squeeze()
        
        return (X, T, Y, potential_Y)  
    
    
    
    
class TWINS_dataset():
    def __init__(self, path = 'Data/Twins/'):
        self.path = path
        self.nProblems = len([file for file in os.listdir(path) if "train" in file])
        
    def getTraining(self, idx = 0):
        
        # Load Training data
        data = np.load(self.path + 'train{}.npz'.format(idx))

        X = data['x']
        T = data['t']
        Y = data['yf']
        # Yfc = data['ycf']

        mu0 = data['mu0'].reshape(-1,1)
        mu1 = data['mu1'].reshape(-1,1)

        potential_Y = np.concatenate( [mu0, mu1], axis=1 )
        
        return (X, T, Y, potential_Y)
    
    
    def getTesting(self, idx = 0):
        
        # Load Testing data
        data = np.load(self.path + 'test{}.npz'.format(idx))

        X = data['x']
        T = data['t']
        Y = data['yf']
        # Yfc = data['ycf']

        mu0 = data['mu0'].reshape(-1,1)
        mu1 = data['mu1'].reshape(-1,1)

        potential_Y = np.concatenate( [mu0, mu1], axis=1 )
        
        return (X, T, Y, potential_Y)  




class IHDP_dataset():
    def __init__(self, path = 'Data/IHDP/'):
        self.path = path
        self.nProblems = 100

        self.Train = np.load(self.path + 'ihdp_npci_1-1000.train.npz')
        self.Test = np.load(self.path + 'ihdp_npci_1-1000.test.npz')

        print('[INFO] Testing data have been successfully imported')
        
    def getTraining(self, idx = 0):
        X  = self.Train['x'][:,:,idx]
        T  = self.Train['t'][:,idx]
        Y  = self.Train['yf'][:,idx]
        Yc = self.Train['ycf'][:,idx]

        mu0 = self.Train['mu0'][:, idx].reshape(-1,1)
        mu1 = self.Train['mu1'][:, idx].reshape(-1,1)

        potential_Y = np.concatenate( [mu0, mu1], axis=1 )
        
        return (X, T, Y, potential_Y)
    
    
    def getTesting(self, idx = 0):
        X  = self.Test['x'][:,:,idx]
        T  = self.Test['t'][:,idx]
        Y  = self.Test['yf'][:,idx]
        Yc = self.Test['ycf'][:,idx]

        mu0 = self.Test['mu0'][:, idx].reshape(-1,1)
        mu1 = self.Test['mu1'][:, idx].reshape(-1,1)

        potential_Y = np.concatenate( [mu0, mu1], axis=1 )
        
        return (X, T, Y, potential_Y)  






def load_and_format_covariates(file_path='x.csv'):
    df = pd.read_csv(file_path, index_col='sample_id',header=0, sep=',')
    return df

def load_treatment_and_outcome(covariates, file_path, standardize=True):
    output = pd.read_csv(file_path, index_col='sample_id', header=0, sep=',')

    dataset = covariates.join(output, how='inner')
    t = dataset['z'].values
    y = dataset['y'].values
    x = dataset.values[:,:-2]
    
#     if standardize:
#         normal_scalar = preprocessing.StandardScaler()
#         x = normal_scalar.fit_transform(x)
    
    return t, y, dataset.index, x

# censoring

class ACIC_dataset():
    def __init__(self, path = 'Data/ACIC/', train_size = 0.9, random_state = 42):
        self.path = path
        self.random_state = random_state
        self.train_size = train_size
        
        
        simulation_files = ["censoring/" + f for f in os.listdir(path+"censoring/") if os.path.isfile(os.path.join(path+"censoring/", f))]+\
                           ["scaling/" + f for f in os.listdir(path+"scaling/") if os.path.isfile(os.path.join(path+"scaling/", f))]
                    
        Filenames = []
        for file in simulation_files:
            cf_suffix      = "_cf"
            file_extension = ".csv"
            
            if file.endswith(cf_suffix + file_extension):
                continue
            
            Filenames += [ file ]
        
        self.Filenames = Filenames
        self.x_raw = load_and_format_covariates(path + 'x.csv')
        self.nProblems = len(Filenames)

        print('[INFO] Number of cases: ', len(Filenames))
        
        
    def createDatasets(self, idx ):
               
        path = self.path + self.Filenames[idx]
        T, Y, sample_id, X = load_treatment_and_outcome(self.x_raw, path)
        
        path = self.path + self.Filenames[idx].split('.csv')[0] + '_cf.csv'
        
        potential_Y = pd.read_csv( path, index_col='sample_id' )
        potential_Y = potential_Y.loc[sample_id][ ['y0', 'y1' ]].values
        
        
        for i in range(Y.shape[0]):
            if (T[i] == 0):
                Y[i] = potential_Y[i,0]
            else:
                Y[i] = potential_Y[i,1]
        
        self.X = X
        self.T = T
        self.Y = Y
        self.potential_Y = potential_Y
        
        
        
        
        self.train_idx, self.test_idx = train_test_split( np.array( range(T.shape[0]) ),
                                                         train_size = self.train_size, 
                                                         random_state = self.random_state)

        
    def getTraining(self, idx=None):
        self.createDatasets(idx)
        idx = self.train_idx
        
        return self.X[idx], self.T[idx], self.Y[idx], self.potential_Y[idx]
    

    def getTesting(self, idx=None):
        idx = self.test_idx
        
        return self.X[idx], self.T[idx], self.Y[idx], self.potential_Y[idx]