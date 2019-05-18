import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def set_options():
    pd.options.display.max_columns = 9999
    SEED=21
    #%matplotlib inline 
    
def save_obj(obj,object_name):
    with open(object_name, 'wb') as f:
        pickle.dump(obj, f)

def load_obj(obj,object_name):
    with open(object_name, 'rb') as f:
        obj = pickle.load(f)

def get_meta(df, target = ['target'], exclude = ['id']):
    data = []
    for f in df.columns:
        #Role
        if f in target:
            role = 'target'
        elif f in exclude:
            role = 'exclude'
        else:
            role = 'input'
        
        #Level
        level=np.nan
        if df[f].dtype == np.dtype('float64'):
            level = 'continuous'
        elif df[f].dtype == np.dtype('int64'):
            level = 'discrete'
        else:
            level = 'nominal'
            
        #Keep
        keep = True
        if f in exclude:
            keep = False
            
        #Dtype
        dtype = df[f].dtype
        
        f_dict = {
            'varname': f,
            'role': role,
            'level': level,
            'keep': keep,
            'dtype': dtype
        }
        data.append(f_dict)
    
    meta = pd.DataFrame(data, columns=['varname', 'role', 'level', 'keep', 'dtype'])
    meta.set_index('varname', inplace=True)

    return meta

def draw_countplots(df, int_cols):
    cont = 0
    for i in int_cols:
        cont+=1
        plt.subplot(len(int_cols)+1,1,cont)
        ax = sns.countplot(df[i])
        ax.set_title(i)
        
def draw_distplots(df, cont_cols):
    cont = 0
    for i in cont_cols:
        cont+=1
        plt.subplot(len(cont_cols)+1,1,cont)
        ax = sns.distplot(df[df[i].notnull()][i])
        ax.set_title(i)
        
def pca_explained_variance(df, features, plot=False):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    z_scaler = StandardScaler()
    pca = PCA()
    pca.fit(z_scaler.fit_transform(df[features].fillna(0)))
    if plot:
        plt.axhline(y=0.8, color='r', linestyle='-')
        plt.semilogy(pca.explained_variance_ratio_, '-o')
        plt.semilogy(pca.explained_variance_ratio_.cumsum(), '--o');
        plt.show()
    return pca.explained_variance_ratio_

def pca_2d(df, features, target, whiten=False):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    z_scaler = StandardScaler()
    pca_2c = PCA(n_components=2, whiten=whiten)
    X_pca_2c = pca_2c.fit_transform(z_scaler.fit_transform(df[features].fillna(0)))
    print(X_pca_2c.shape)
    plt.scatter(X_pca_2c[:,0], X_pca_2c[:,1], c=df[target].values.ravel(), alpha=0.8, 
                s=60, marker='o', edgecolors='white')
    plt.show()
    return pca_2c.explained_variance_ratio_.sum() 
