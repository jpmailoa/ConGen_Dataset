import os
import pandas as pds
import numpy as np
import pickle
import rdkit
from rdkit import Chem
import matplotlib.pyplot as plt

def load_training_data(path):
    smiles = []
    data = pds.read_csv( path )
    names = [name for name in data]
    assert names[0] == 'SMILES'
    for j in range(len(data['SMILES'])):
        try:
            new_smiles = Chem.MolToSmiles( Chem.MolFromSmiles( data['SMILES'][j] ))
            smiles.append( new_smiles )
        except:
            continue
    return list(set(smiles))

def load_gen_smiles(path):
    lines = open(path,'r').read().strip().split('\n')
    new_smiles, MolWt, LogP, QED = [], [], [], []
    def insert_dummy():
        new_smiles.append('invalid')
        MolWt.append( 0 )
        LogP.append( 0 )
        QED.append( 0 )
    for line in lines:
        words = line.split()
        if len(words)==5:
            smiles = words[1][1:-2]
            try:
                new_smiles.append( Chem.MolToSmiles( Chem.MolFromSmiles( smiles )) )
                MolWt.append( float(words[2][1:-1]) )
                LogP.append( float(words[3][:-1]) )
                QED.append( float(words[4][:-2]) )
            except:
                insert_dummy()
        else:
            insert_dummy()
    return new_smiles, np.array( [MolWt,LogP,QED] ).T

def collect_singles():
    single_smiles, single_prop = [], []
    smiles, prop = load_gen_smiles('out_single_MolWt')
    single_smiles += smiles
    single_prop.append( prop )
    smiles, prop = load_gen_smiles('out_single_LogP')
    single_smiles += smiles
    single_prop.append( prop )
    smiles, prop = load_gen_smiles('out_single_QED')
    single_smiles += smiles
    single_prop.append( prop )
    single_prop = np.concatenate(single_prop, axis=0)
    return single_smiles, single_prop

def collect_multi():
    smiles, prop = load_gen_smiles('out_multi')
    return smiles, prop

def count_constrained_novel(train_smiles, gen_smiles, gen_prop, tolerance):
    tolerance = abs(tolerance)
    prop_ref = np.array([250.0, 2.5, 0.55])
    gen_count = [i for i in range(len(gen_smiles)+1)]
    gen_novel_valid = [0]
    gen_novel = [0]
    #new_smiles = []
    new_smiles = dict()
    temp = dict()
    for smiles in train_smiles:
        temp[ smiles ] = True
    train_smiles = temp
    del temp
    fail_constraint = np.array([0,0,0])
    within_train = 0
    for i, smiles in enumerate(gen_smiles):
        if i%1000==0: print(i)
        if smiles == 'invalid':
            gen_novel_valid.append( gen_novel_valid[-1] )
            gen_novel.append( gen_novel[-1] )
        elif (smiles in new_smiles) or (smiles in train_smiles):
            gen_novel_valid.append( gen_novel_valid[-1] )
            gen_novel.append( gen_novel[-1] )
            if smiles in train_smiles:
                within_train += 1
        else:
            #new_smiles.append( smiles )
            new_smiles[ smiles ] = True
            gen_novel.append( gen_novel[-1] + 1 )
            constraint = np.abs(gen_prop[i] - prop_ref)/prop_ref < tolerance
            if np.all(constraint):
                gen_novel_valid.append( gen_novel_valid[-1] + 1 )
            else:
                gen_novel_valid.append( gen_novel_valid[-1] )
            fail_constraint += ~constraint
    print('Unique:', len(new_smiles))
    print('Fail properties:', fail_constraint)
    print('Within training set:', within_train)
    return gen_count, gen_novel, gen_novel_valid, len(new_smiles)
    
train_smiles = load_training_data('data/8117_291883_ZINC_310k_delabel_SSVAE.csv')
pickle.dump(train_smiles, open('train_smiles.pkl','wb'))
##train_smiles = pickle.load(open('train_smiles.pkl','rb'))

SSVAE_smiles, SSVAE_prop = load_gen_smiles('baseSSVAE_gen_log_multi.txt')
ConGen_smiles, ConGen_prop = load_gen_smiles('ConGen_gen_log_multi.txt')

SSVAE_count_20, SSVAE_novel_20, SSVAE_novel_valid_20, SSVAE_total_novel_20 = count_constrained_novel(train_smiles, SSVAE_smiles, SSVAE_prop, 0.2)
pickle.dump([SSVAE_count_20, SSVAE_novel_20, SSVAE_novel_valid_20, SSVAE_total_novel_20], open('SSVAE_20.pkl','wb'))
SSVAE_count_10, SSVAE_novel_10, SSVAE_novel_valid_10, SSVAE_total_novel_10 = count_constrained_novel(train_smiles, SSVAE_smiles, SSVAE_prop, 0.1)
pickle.dump([SSVAE_count_10, SSVAE_novel_10, SSVAE_novel_valid_10, SSVAE_total_novel_10], open('SSVAE_10.pkl','wb'))
##SSVAE_count_20, SSVAE_novel_20, SSVAE_novel_valid_20, SSVAE_total_novel_20 = pickle.load(open('SSVAE_20.pkl','rb'))
##SSVAE_count_10, SSVAE_novel_10, SSVAE_novel_valid_10, SSVAE_total_novel_10 = pickle.load(open('SSVAE_10.pkl','rb'))

ConGen_count_20, ConGen_novel_20, ConGen_novel_valid_20, ConGen_total_novel_20 = count_constrained_novel(train_smiles, ConGen_smiles, ConGen_prop, 0.2)
pickle.dump([ConGen_count_20, ConGen_novel_20, ConGen_novel_valid_20, ConGen_total_novel_20], open('ConGen_20.pkl','wb'))
ConGen_count_10, ConGen_novel_10, ConGen_novel_valid_10, ConGen_total_novel_10 = count_constrained_novel(train_smiles, ConGen_smiles, ConGen_prop, 0.1)
pickle.dump([ConGen_count_10, ConGen_novel_10, ConGen_novel_valid_10, ConGen_total_novel_10], open('ConGen_10.pkl','wb'))
##ConGen_count_20, ConGen_novel_20, ConGen_novel_valid_20, ConGen_total_novel_20 = pickle.load(open('ConGen_20.pkl','rb'))
##ConGen_count_10, ConGen_novel_10, ConGen_novel_valid_10, ConGen_total_novel_10 = pickle.load(open('ConGen_10.pkl','rb'))

print('Acceptance rate SSVAE-20:', SSVAE_novel_valid_20[-1]/SSVAE_novel_20[-1])
print('Acceptance rate SSVAE-10:', SSVAE_novel_valid_10[-1]/SSVAE_novel_10[-1])
print('Acceptance rate ConGen-20:', ConGen_novel_valid_20[-1]/ConGen_novel_20[-1])
print('Acceptance rate ConGen-10:', ConGen_novel_valid_10[-1]/ConGen_novel_10[-1])

plt.figure(0, figsize=(6,5))
plt.loglog(ConGen_novel_20[1:], ConGen_novel_valid_20[1:],'b')
plt.plot(ConGen_novel_10[1:], ConGen_novel_valid_10[1:],'b--')
plt.plot(SSVAE_novel_20[1:], SSVAE_novel_valid_20[1:],'r')
plt.plot(SSVAE_novel_10[1:], SSVAE_novel_valid_10[1:],'r--')
plt.xlabel('Unique novel molecules', fontsize=18)
plt.ylabel('Accepted molecules', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(['ConGen multi - 20% tol.','ConGen multi - 10% tol.','SSVAE multi - 20% tol.','SSVAE multi - 10% tol.'], fontsize=14)
plt.axis([1,10000,1,10000])
plt.tight_layout()
plt.show()

