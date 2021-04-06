import os
import pandas as pd
import numpy as np

def load_csv(filepath, delimiter=',', class_col='', dataset_name='', print_summary=False):
    """[summary]

    Args:
        filepath (str): filepath of the CSV
        delimiter (str, optional): CSV delimiter. Defaults to ''.
        class_col (str, optional): name of the class column. If empty, last column is used as class. Defaults to ''.
        dataset_name (str, optional): name of the dataset, to give to the DataFrame. Defaults to ','.
        print_summary (bool, optional): Print a summary of the dataset. Defaults to False.

    Returns:
        tuple: containing a DataFrame of examples features and a Series of classes
    """
    # Load CSV dataset as pandas table
    df = pd.read_csv(filepath, delimiter=delimiter)
    if len(dataset_name):
        df.dataframeName = dataset_name
    
    # Class column defaults to last column
    if not len(class_col):
        class_col = df.keys()[-1]
    
    # Dataset summary
    if print_summary:
        for c in df.columns:
            if c!=class_col:
                print(df.groupby([c, class_col]).size())
                print('------------------')
        df[class_col].value_counts()
        
    # Separate data and class
    df_class = df[class_col]
    df.drop(class_col, 1, inplace=True)
    
    return df, df_class
    
def load_tennis(print_summary=False):
    """Load play tennis dataset
    https://www.kaggle.com/fredericobreno/play-tennis
    
    Args:
        print_summary (bool, optional): Print a summary of the dataset. Defaults to False.

    Returns:
        tuple: containing a DataFrame of examples features and a Series of classes
    """
    # Load CSV dataset as pandas table
    df = pd.read_csv('datasets/play_tennis.csv', delimiter=',')
    df.dataframeName = 'Play Tennis'
    
    # Remove day column (not a feature)
    df.drop('day', 1, inplace=True)
    
    # Dataset summary
    if print_summary:
        for c in df.columns:
            if c!='play':
                print(df.groupby([c, 'play']).size())
                print('------------------')
        df['play'].value_counts()
    
    # Separate data and class
    df_class = df['play']
    df.drop('play', 1, inplace=True)
    
    return df, df_class

def load_lenses(print_summary=False):
    """ Load lenses dataset
    https://archive.ics.uci.edu/ml/datasets/lenses
    
    Attribute Information:
    -- 3 Classes
    1 : the patient should be fitted with hard contact lenses,
    2 : the patient should be fitted with soft contact lenses,
    3 : the patient should not be fitted with contact lenses.

    1. age of the patient: (1) young, (2) pre-presbyopic, (3) presbyopic
    2. spectacle prescription: (1) myope, (2) hypermetrope
    3. astigmatic: (1) no, (2) yes
    4. tear production rate: (1) reduced, (2) normal
    
    Args:
        print_summary (bool, optional): Print a summary of the dataset. Defaults to False.
        
    Returns:
        tuple: containing a DataFrame of examples features and a Series of classes
    """
    # Load DATA dataset as pandas table
    lenses_header = ['age', 'deficiency', 'astigmatic', 'tear production', 'recommendation']
    df = pd.read_table('datasets/lenses.data', delimiter='\s+', index_col=0, names=lenses_header)
    df.reset_index(drop=True, inplace=True)    # start index at 0
    df.dataframeName = 'Lenses'
    
    # Replace attributes
    replace_dicts = {'age': {1: 'young', 2: 'pre-presbyopic', 3:'presbyopic'},
                     'deficiency': {1: 'myope', 2: 'hypermetrope'},
                     'astigmatic': {1: 'no', 2: 'yes'},
                     'tear production': {1: 'reduced', 2: 'normal'},
                     'recommendation': {1: 'hard', 2: 'soft', 3: 'none'}}
    
    for k in replace_dicts.keys():
        df[k].replace(replace_dicts[k], inplace=True)
    
    # Dataset summary
    if print_summary:
        for c in df.columns:
            if c!='recommendation':
                print(df.groupby([c, 'recommendation']).size())
                print('------------------')
        df['recommendation'].value_counts()
        
    # Separate data and class
    df_class = df['recommendation']
    df.drop('recommendation', 1, inplace=True)
    
    return df, df_class

def load_mammographic_mass(print_summary=False):
    """ Load mammographic mass dataset
    https://archive.ics.uci.edu/ml/datasets/Mammographic+Mass
    
    Attribute Information:
    1. BI-RADS assessment: 1 to 5 (ordinal, non-predictive!)
    2. Age: patient's age in years (integer)
    3. Shape: mass shape: round=1 oval=2 lobular=3 irregular=4 (nominal)
    4. Margin: mass margin: circumscribed=1 microlobulated=2 obscured=3 ill-defined=4 spiculated=5 (nominal)
    5. Density: mass density high=1 iso=2 low=3 fat-containing=4 (ordinal)
    6. Severity: benign=0 or malignant=1 (binominal, goal field!)
    
    Args:
        print_summary (bool, optional): Print a summary of the dataset. Defaults to False.
        
    Returns:
        tuple: containing a DataFrame of examples features and a Series of classes
    """
    # Load DATA dataset as pandas table
    header = ['bi-rads', 'age', 'shape', 'margin', 'density', 'severity']
    df = pd.read_table('datasets/mammographic_masses.data', index_col=False, delimiter=',', names=header)
    df.drop('bi-rads', 1, inplace=True)     # remove non-predictive column
    df.dataframeName = 'Mammographic Mass'

    # Remove examples with missing values
    df.replace({'?':np.nan}, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Replace attributes
    replace_dicts = {'shape': {'1': 'round', '2': 'oval', '3':'lobular', '4':'irregular'},
                     'margin': {'1': 'circumscribed', '2': 'microlobulated', '3':'obscured', '4':'ill-defined', '5':'spiculated'},
                     'density': {'1': 'high', '2': 'iso', '3':'low', '4':'fat-containing'},
                     'severity': {0: 'benign', 1: 'malignant'}}
    
    for k in replace_dicts.keys():
        df[k].replace(replace_dicts[k], inplace=True)

    df['age'] = df['age'].astype(np.int16)       # convert str to int
    
    class_col = 'severity'
    
    # Dataset summary
    if print_summary:
        for c in df.columns:
            if c!=class_col:
                print(df.groupby([c, class_col]).size())
                print('------------------')
        df[class_col].value_counts()
        
    # Separate data and class
    df_class = df[class_col]
    df.drop(class_col, 1, inplace=True)
    
    return df, df_class

