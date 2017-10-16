import glob
import pickle

rfiles = glob.glob('./logs/reward_*.pkl')

print(rfiles)
with open(rfiles[-1], 'rb') as f:
    x = pickle.load(f)
    print(x)
