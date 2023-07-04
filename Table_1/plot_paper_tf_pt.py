import matplotlib.pyplot as plt
import numpy as np

def load_curves(log_file):
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

trn_pt, val_pt, test_pt, uncond_pt, cond_pt = load_curves('paper_code_v1_pt/pt_log')
trn_tf, val_tf, test_tf, uncond_tf, cond_tf = load_curves('paper_code_v1_tf/tf_log')

# Training
plt.figure(0, figsize=[5,5])
plt.semilogy(trn_pt, 'b', linewidth=2)
plt.semilogy(trn_tf, 'r', linewidth=2)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Cost', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Training', fontsize=14)
plt.legend(['PyTorch','TensorFlow'], fontsize=14)
plt.tight_layout()

# Validation
plt.figure(1, figsize=[5,5])
plt.semilogy(val_pt, 'b', linewidth=2)
plt.semilogy(val_tf, 'r', linewidth=2)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Cost', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Validation', fontsize=14)
plt.legend(['ConGen','SSVAE'], fontsize=14)
plt.tight_layout()

# Test
print('Test MAE ConGen:\t',test_pt)
print('Test MAE SSVAE:\t',test_tf)

# Unconditional sampling
s_pt = sampling_statistics(uncond_pt)
s_tf = sampling_statistics(uncond_tf)
print('Unconditional sampling mean ConGen:\t',s_pt[0])
print('Unconditional sampling std ConGen:\t',s_pt[1])
print('Unconditional sampling mean SSVAE:\t',s_tf[0])
print('Unconditional sampling std SSVAE:\t',s_tf[1])

# Conditional sampling
s_pt = sampling_statistics(cond_pt)
s_tf = sampling_statistics(cond_tf)
print('Conditional sampling mean ConGen:\t',s_pt[0])
print('Conditional sampling std ConGen:\t',s_pt[1])
print('Conditional sampling mean SSVAE:\t',s_tf[0])
print('Conditional sampling std SSVAE:\t',s_tf[1])

plt.show()
