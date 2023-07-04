import matplotlib.pyplot as plt
import numpy as np
import json

def load_curves_rnn(ipynb_file):
    trn, val = [], []
    test = []
    uncond, cond = dict(), dict()
    data = json.load(open(ipynb_file,'r'))
    data = data['cells']
    trn_lines = data[6]['outputs'][0]['text']
    test_MAE_lines = data[10]['outputs'][0]['text']
    gen_uncond_lines = data[12]['outputs'][0]['text']
    gen_cond_lines = data[14]['outputs'][0]['text']

    for line in trn_lines:
        line = line.strip()
        if "['Training', 'cost_trn'" in line:
            cost = float(line.split()[-1][:-1])
            trn.append(cost)
        elif "['Validation', 'cost_val'" in line:
            cost = float(line.split()[-1][:-1])
            val.append(cost)

    for line in test_MAE_lines:
        words = line.strip().split()
        if len(words)==2:
            MAE = float(words[-1][:-1])
            test.append(MAE)

    for line in gen_uncond_lines:
        words = line.strip().split()
        if len(words)==5:
            idx = int(words[0][1:-1])
            smiles = str(words[1][1:-2])
            prop1 = float(words[2][1:-1])
            prop2 = float(words[3][0:-1])
            prop3 = float(words[4][0:-2])
            uncond[ idx ] = [smiles, [prop1,prop2,prop3] ]
    
    for line in gen_cond_lines:
        words = line.strip().split()
        if len(words)==5:
            idx = int(words[0][1:-1])
            smiles = str(words[1][1:-2])
            prop1 = float(words[2][1:-1])
            prop2 = float(words[3][0:-1])
            prop3 = float(words[4][0:-2])
            cond[ idx ] = [smiles, [prop1,prop2,prop3] ]
            
    return trn, val, test, uncond, cond

def load_curves_bert(log_file):
    trn, val = [], []
    test = []
    uncond, cond = dict(), dict()
    with open(log_file,'r') as f:
        line = f.readline().strip()
        while line:
            if "['Training', 'cost_trn'" in line:
                cost = float(line.split()[-1][:-1])
                trn.append(cost)
            elif "['Validation', 'cost_val'" in line:
                cost = float(line.split()[-1][:-1])
                val.append(cost)
            elif line.startswith('[') and line.endswith(']'):
                words = line.split()
                if len(words)==2:
                    MAE = float(words[-1][:-1])
                    test.append(MAE)
                elif len(words)==5:
                    idx = int(words[0][1:-1])
                    if idx in uncond:   to_use = cond
                    else:               to_use = uncond
                    smiles = str(words[1][1:-2])
                    prop1 = float(words[2][1:-1])
                    prop2 = float(words[3][0:-1])
                    prop3 = float(words[4][0:-2])
                    to_use[ idx ] = [smiles, [prop1,prop2,prop3] ]
            line = f.readline().strip()
    return trn, val, test, uncond, cond

def sampling_statistics(dict_out):
    out = []
    for idx, [smiles, prop] in dict_out.items():
        out.append(prop)
    out = np.array(out)
    return np.mean(out,axis=0), np.std(out,axis=0)

trn_rnn, val_rnn, test_rnn, uncond_rnn, cond_rnn = load_curves_rnn('paper_ConGen_table2_oriRNN/train_generator.ipynb')
trn_bert, val_bert, test_bert, uncond_bert, cond_bert = load_curves_bert('paper_ConGen_table2_trfBERT/bert_log')

# Training
plt.figure(0, figsize=[5,5])
plt.semilogy(trn_rnn, 'b', linewidth=2)
plt.semilogy(trn_bert, 'r', linewidth=2)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Cost', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Training', fontsize=14)
plt.legend(['ConGen RNN','ConGen BERT'], fontsize=14)
plt.tight_layout()

# Validation
plt.figure(1, figsize=[5,5])
plt.semilogy(val_rnn, 'b', linewidth=2)
plt.semilogy(val_bert, 'r', linewidth=2)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Cost', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Validation', fontsize=14)
plt.legend(['ConGen RNN','ConGen BERT'], fontsize=14)
plt.tight_layout()

# Test
print('Test MAE ConGen RNN:\t',test_rnn)
print('Test MAE ConGen BERT:\t',test_bert)

# Unconditional sampling
s_rnn = sampling_statistics(uncond_rnn)
s_bert = sampling_statistics(uncond_bert)
print('Unconditional sampling mean ConGen RNN:\t',s_rnn[0])
print('Unconditional sampling std ConGen RNN:\t',s_rnn[1])
print('Unconditional sampling mean ConGen BERT:\t',s_bert[0])
print('Unconditional sampling std ConGen BERT:\t',s_bert[1])

# Conditional sampling
s_rnn = sampling_statistics(cond_rnn)
s_bert = sampling_statistics(cond_bert)
print('Conditional sampling mean ConGen RNN:\t',s_rnn[0])
print('Conditional sampling std ConGen RNN:\t',s_rnn[1])
print('Conditional sampling mean Congen BERT:\t',s_bert[0])
print('Conditional sampling std ConGen BERT:\t',s_bert[1])

plt.show()
