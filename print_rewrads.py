import glob
import pickle
import numpy as np

rfiles = glob.glob('./logs/net_*.pkl')

print(rfiles)

for rf in rfiles:
    with open(rf, 'rb') as f:
        x = pickle.load(f)

        max_acc = (0, 0, 0)
        for k, v in x.items():
            if v['accuracy'][0] > max_acc[1]:
                max_acc = (k, v['accuracy'][0], v['code'])
        acc = [x[xx]['accuracy'][0] for xx in x]

        print(max(acc), np.array(acc).mean(), np.array(acc).std())
        #print(max_acc)
