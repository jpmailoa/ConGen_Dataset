import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import pickle
import pandas as pds
from sklearn.preprocessing import StandardScaler

def get_property(smi):

    try:
        mol=Chem.MolFromSmiles(smi) 
        property = [Descriptors.ExactMolWt(mol), Descriptors.MolLogP(mol), QED.qed(mol)]
        
    except:
        property = 'invalid'
           
    return property
    

def canonocalize(smi):

    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))
