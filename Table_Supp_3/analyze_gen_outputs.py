import numpy as np

tags = 'MolWt', 'LogP', 'IE'

overall_out = []
for tag in tags:
    lines = open('gen_log_single_'+tag+'.txt','r').read().split('\n')[0:10]
    out = []
    for line in lines:
        words = line.strip().split()
        MolWt = float(words[2][1:-1])
        LogP = float(words[3][0:-1])
        QED = float(words[4][0:-2])
        out.append([MolWt, LogP, QED])
    out = np.array(out)
    overall_out = []
    print(tag)
    print('MolWt, LogP, QED mean:', np.mean(out, axis=0))
    print('MolWt, LogP, QED std:', np.std(out, axis=0))

overall_out = np.concatenate(overall_out, axis=0)
print('Overall')
print('MolWt, LogP, QED mean:', np.mean(overall_out, axis=0))
print('MolWt, LogP, QED std:', np.std(overall_out, axis=0))