# NAS Replication

## Start Server
On Server CC11, start a tmux
```
cd zhoubin/nas
python main.py
```

## Start Clients
```
send-all "cd zhoubin/nas; ./multi_client.sh 192.168.0.11 0 1 2 3 4 5 6 7"
```

## Restart Clients
```
send-all "rm -rf ~/zhoubin/nas/logs"
send-all "tmux kill-session -t client"
send-all "cd zhoubin/nas; ./multi_client.sh 192.168.0.11 0 1 2 3 4 5 6 7"

```
## Check Results
Result dictonary is saved under
```
nas/logs/step_%05d.pkl
```

For example, if you want to plot results trained till 3000 steps:

```python
import pickle
with open('step_03000.pkl', 'rb') as f:
    data = pickle.load(f)

accuracy = []
max_acc = 0
for idx, tab in data.items():
    accuracy.append(tab['acc'])

plt.figure()
plt.plot(accuracy)
plt.figure()
plt.plot(np.divide(np.cumsum(accuracy), list(range(1,3001))))
plt.show()

code_acc_list = []
for idx, tab in data.items():
    code_acc_list.append((idx, tab['acc'], tab['code']))


sorted_code_acc_list = sorted(code_acc_list, key=lambda x: x[1])
print(sorted_code_acc_list[-1])
for x in sorted_code_acc_list[-100:]:
    model = NASModel(code=x[2], N=5)
    sym = model.build_cifar_network()
    sym.save('symbols/sym_%06d.json' % x[0])
```
