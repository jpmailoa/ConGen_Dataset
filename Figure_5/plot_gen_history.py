import pandas as pds
from preprocessing import canonocalize
import pickle, json
import os
import numpy as np
import matplotlib.pyplot as plt

def load_csv(csv_name):
    assert csv_name.endswith('.csv')
    data = pds.read_csv( csv_name )
    names = [name for name in data]
    assert names[0] == 'SMILES'
    smiles = []
    for j in range(len(data['SMILES'])):
        smiles.append(canonocalize(data['SMILES'][j]))
    return smiles

def generate_train_smiles():
    csv_data = ['data/paper_MP_IE_EA.csv',
                'data/paper_MP_clean_canonize_cut.csv',
                'data/paper_clean_viscosity.csv',
                'data/paper_ZINC_310k.csv',
                'data/paper_pubchem_fluorocarbon.csv']
    smiles_list = []
    for csv in csv_data:
        smiles_list += load_csv( csv )
        print(len(smiles_list))
    unique_smiles_list = list(set(smiles_list))
    print('Unique:', len(unique_smiles_list))
    pickle.dump(unique_smiles_list, open('train_smiles.pkl','wb'))
    unique_smiles_list = pickle.load(open('train_smiles.pkl','rb'))
    return unique_smiles_list

def known_LHCE_smiles(train_smiles):
    literature_LHCE = [
                       # in training dataset
                       'C(C(F)(F)F)OCC(F)(F)F', #BTFE found
                       'CC1=CC(=CC=C1)F', #m-fluorotoluene
                       'B(OCC(F)(F)F)(OCC(F)(F)F)OCC(F)(F)F', #TFEB
                       
                       # not in training dataset
                       'C(C(C(F)F)(F)F)OC(C(F)F)(F)F', #TTE found
                       'C(C(F)(F)F)OC(OCC(F)(F)F)OCC(F)(F)F', #TFEO
                       'COC(C(F)(F)F)C(F)(F)F', #HFPM found
                       'C1=CC=C(C=C1)C(F)(F)F', #benzotrifluoride
                       'C(C(C(C(C(F)F)(F)F)(F)F)(F)F)OC(C(F)F)(F)F', #1H,1H,5H-octafluoropentyl 1,1,2,2-tetrafluoroethyl ether
                       'C(C(F)(F)F)OC(C(F)F)(F)F', #2,2,2-trifluoroethyl 1,1,2,2-tetrafluoroethyl ether found
                       'C(C(F)(F)F)OC(=O)OCC(F)(F)F', #BTFEC found
                       'CCOP1(=NP(=NP(=N1)(F)F)(F)F)F', #PFPN
                       'C(C(F)(F)F)OS(=O)(=O)C(F)(F)F'] #TFEOTf
    known_LHCE = [canonocalize(smiles) for smiles in literature_LHCE]
    for smiles in known_LHCE:
        print(smiles,'in training database:',str(smiles in train_smiles))
    print('')
    return known_LHCE

def parse_generated_smiles(train_smiles, known_LHCE):
    unique_smiles = []
    molwt = []
    nF = []
    nO = []
    nP = []
    nS = []
    with open('gen_log.txt','r') as f:
        lines = f.read().strip().split('\n')
        gen_count = [i for i in range(len(lines)+1)]
        unique_count = [0]
        valid_count = 0
        for line in lines:
            words = line.split()
            if len(words)==5:
                smiles = words[1][1:-2]
                smiles = canonocalize(smiles)
                if smiles in unique_smiles:
                    unique_count.append( unique_count[-1] )
                else:
                    molwt.append( float(words[2][1:-1]) )
                    nF.append( smiles.count('F')+smiles.count('f') )
                    nO.append( smiles.count('O')+smiles.count('o') )
                    nP.append( smiles.count('P')+smiles.count('p') )
                    nS.append( smiles.count('S')+smiles.count('s') )
                    unique_smiles.append( smiles )
                    unique_count.append( unique_count[-1]+1 )
                valid_count += 1
            else:
                unique_count.append( unique_count[-1] )
    print(valid_count,'valid from all generated:',len(lines))
    print(len(unique_smiles),'unique from the valid generated:',valid_count)
    print('Gen. MolWt:', np.mean(np.array(molwt)), '+/-',  np.std(np.array(molwt)))
    print('Gen. n_F:', np.mean(np.array(nF)), '+/-',  np.std(np.array(nF)))
    print('Gen. n_O:', np.mean(np.array(nO)), '+/-',  np.std(np.array(nO)))
    print('Gen. n_P:', np.mean(np.array(nP)), '+/-',  np.std(np.array(nP)))
    print('Gen. n_S:', np.mean(np.array(nS)), '+/-',  np.std(np.array(nS)))
    print('Mol with P:', (np.array(nP)!=0).sum(), 'out of',len(nP))
    print('Mol with S:', (np.array(nS)!=0).sum(), 'out of',len(nS))
    print('Mol with F>=9:', (np.array(nF)>=9).sum(), 'out of',len(nF))
    print('')
    found_mols = []
    for smiles in known_LHCE:
        found = smiles in unique_smiles
        print(smiles, 'in new generated set:', str(found))
        if found:
            y = unique_smiles.index(smiles) + 1
            x = gen_count[ unique_count.index(y) ]
            found_mols.append([smiles,x,y])
            print(smiles,'found at:',str((x,y)))
    print('')

    fts = 18
    # MolWt
    plt.figure(0, figsize=(6,5))
    plt.hist(molwt, bins=[i*10-5 for i in range(10,46)], color='seagreen', alpha=0.5)
    for i in range(21):
        x = i*10 + 150
        plt.plot([x,x],[0,2500],'k--')
    plt.xlabel('Mol.Wt (Da)', fontsize=fts)
    plt.ylabel('Count', fontsize=fts)
    plt.xticks(fontsize=fts)
    plt.yticks(fontsize=fts)
    plt.axis([100,450,0,2500])
    plt.tight_layout()

    # nF
    plt.figure(1, figsize=(6,5))
    plt.hist(nF, bins=[i-0.5 for i in range(0,14)], color='seagreen', alpha=0.5)
    for i in range(4,10):
        x = i
        plt.plot([x,x],[0,8000],'k--')
    plt.xlabel(r'$n_F$', fontsize=fts)
    plt.ylabel('Count', fontsize=fts)
    plt.xticks(fontsize=fts)
    plt.yticks(fontsize=fts)
    plt.axis([0,14,0,8000])
    plt.tight_layout()

    # nO
    plt.figure(2, figsize=(6,5))
    plt.hist(nO, bins=[i-0.5 for i in range(0,6)], color='seagreen', alpha=0.5)
    for i in range(1,4):
        x = i
        plt.plot([x,x],[0,12000],'k--')
    plt.xlabel(r'$n_O$', fontsize=fts)
    plt.ylabel('Count', fontsize=fts)
    plt.xticks(fontsize=fts)
    plt.yticks(fontsize=fts)
    plt.axis([0,6,0,12000])
    plt.tight_layout()
    
    return gen_count, unique_count, found_mols

def plot_gen(gen_count, unique_count, found_mols):
    plt.figure(3, figsize=(6,5))
    plt.plot(gen_count, unique_count, 'seagreen', linewidth=2)
    for entry in found_mols:
        plt.scatter([entry[1]],[entry[2]],color='seagreen')
        plt.plot([entry[1],entry[1]],[0,entry[2]], '--k')
        plt.plot([0,entry[1]],[entry[2],entry[2]], '--k')
    plt.xlabel('Generation Count', fontsize=18)
    plt.ylabel('Unique Count', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.plot([0,unique_count[-1]],[0,unique_count[-1]], '--k', alpha=0.3)
    plt.axis([0,gen_count[-1],0,unique_count[-1]])
    plt.tight_layout()
    plt.show()
    return

#train_smiles = generate_train_smiles()
train_smiles = pickle.load(open('train_smiles.pkl','rb'))
known_LHCE = known_LHCE_smiles(train_smiles)
gen_count, unique_count, found_mols = parse_generated_smiles(train_smiles, known_LHCE)
plot_gen(gen_count, unique_count, found_mols)

