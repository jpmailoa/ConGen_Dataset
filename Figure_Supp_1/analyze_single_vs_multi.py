import os
import pandas as pds
import numpy as np
import pickle
import rdkit
from rdkit import Chem
import matplotlib.pyplot as plt

def load_training_data(folder):
    csvs = os.listdir(folder)
    smiles = []
    for csv in csvs:
        data = pds.read_csv( os.path.join(folder,csv) )
        names = [name for name in data]
        assert names[0] == 'SMILES'
        for j in range(len(data['SMILES'])):
            try:
                new_smiles = Chem.MolToSmiles( Chem.MolFromSmiles( data['SMILES'][j] ))
                smiles.append( new_smiles )
            except:
                continue
    return list(set(smiles))

def load_gen_smiles(folder):
    lines = open(os.path.join(folder,'gen_log.txt'),'r').read().strip().split('\n')
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
    for i, smiles in enumerate(gen_smiles):
        if i%1000==0: print(i)
        if smiles == 'invalid':
            gen_novel_valid.append( gen_novel_valid[-1] )
            gen_novel.append( gen_novel[-1] )
        elif (smiles in new_smiles) or (smiles in train_smiles):
            gen_novel_valid.append( gen_novel_valid[-1] )
            gen_novel.append( gen_novel[-1] )
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
    print('Fail properties:', fail_constraint)
    return gen_count, gen_novel, gen_novel_valid, len(new_smiles)
    
#train_smiles = load_training_data('training_data')
#pickle.dump(train_smiles, open('train_smiles.pkl','wb'))

#single_smiles, single_prop = collect_singles()
#pickle.dump([single_smiles, single_prop], open('single_gen.pkl','wb'))

#multi_smiles, multi_prop = collect_multi()
#pickle.dump([multi_smiles, multi_prop], open('multi_gen.pkl','wb'))



train_smiles = pickle.load(open('train_smiles.pkl','rb'))

single_smiles, single_prop = pickle.load(open('single_gen.pkl','rb'))
#single_count_30, single_novel_30, single_novel_valid_30, single_total_novel_30 = count_constrained_novel(train_smiles, single_smiles, single_prop, 0.3)
#pickle.dump([single_count_30, single_novel_30, single_novel_valid_30, single_total_novel_30], open('single_30.pkl','wb'))
#single_count_20, single_novel_20, single_novel_valid_20, single_total_novel_20 = count_constrained_novel(train_smiles, single_smiles, single_prop, 0.2)
#pickle.dump([single_count_20, single_novel_20, single_novel_valid_20, single_total_novel_20], open('single_20.pkl','wb'))
#single_count_10, single_novel_10, single_novel_valid_10, single_total_novel_10 = count_constrained_novel(train_smiles, single_smiles, single_prop, 0.1)
#pickle.dump([single_count_10, single_novel_10, single_novel_valid_10, single_total_novel_10], open('single_10.pkl','wb'))
single_count_20, single_novel_20, single_novel_valid_20, single_total_novel_20 = pickle.load(open('single_20.pkl','rb'))
single_count_10, single_novel_10, single_novel_valid_10, single_total_novel_10 = pickle.load(open('single_10.pkl','rb'))

multi_smiles, multi_prop = pickle.load(open('multi_gen.pkl','rb'))
#multi_count_30, multi_novel_30, multi_novel_valid_30, multi_total_novel_30 = count_constrained_novel(train_smiles, multi_smiles, multi_prop, 0.3)
#pickle.dump([multi_count_30, multi_novel_30, multi_novel_valid_30, multi_total_novel_30], open('multi_30.pkl','wb'))
#multi_count_20, multi_novel_20, multi_novel_valid_20, multi_total_novel_20 = count_constrained_novel(train_smiles, multi_smiles, multi_prop, 0.2)
#pickle.dump([multi_count_20, multi_novel_20, multi_novel_valid_20, multi_total_novel_20], open('multi_20.pkl','wb'))
#multi_count_10, multi_novel_10, multi_novel_valid_10, multi_total_novel_10 = count_constrained_novel(train_smiles, multi_smiles, multi_prop, 0.1)
#pickle.dump([multi_count_10, multi_novel_10, multi_novel_valid_10, multi_total_novel_10], open('multi_10.pkl','wb'))
multi_count_20, multi_novel_20, multi_novel_valid_20, multi_total_novel_20 = pickle.load(open('multi_20.pkl','rb'))
multi_count_10, multi_novel_10, multi_novel_valid_10, multi_total_novel_10 = pickle.load(open('multi_10.pkl','rb'))

print('Acceptance rate multi-20:', multi_novel_valid_20[-1]/multi_novel_20[-1])
print('Acceptance rate multi-10:', multi_novel_valid_10[-1]/multi_novel_10[-1])
print('Acceptance rate single-20:', single_novel_valid_20[-1]/single_novel_20[-1])
print('Acceptance rate single-10:', single_novel_valid_10[-1]/single_novel_10[-1])

plt.figure(0, figsize=(6,5))
#plt.plot(multi_novel_30, multi_novel_valid_30,'b')
#plt.plot(single_novel_30, single_novel_valid_30,'r')
plt.plot(multi_novel_20, multi_novel_valid_20,'b')
plt.plot(multi_novel_10, multi_novel_valid_10,'b--')
plt.plot(single_novel_20, single_novel_valid_20,'r')
plt.plot(single_novel_10, single_novel_valid_10,'r--')
plt.xlabel('Unique molecules', fontsize=18)
plt.ylabel('Accepted molecules', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
plt.legend(['Multi - 20% tol.','Multi - 10% tol.','Single - 20% tol.','Single - 10% tol.'], fontsize=14)
plt.axis([0,70000,0,7000])
plt.tight_layout()
plt.show()

