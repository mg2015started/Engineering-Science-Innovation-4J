import numpy as np

def hardlim(x):
    if x >= 0:
        return 1
    else:
        return 0

w = np.array([0,0])
b = 0

node = np.array([[1,-1],[-1,-1],[0,0],[1,0]])
labels = np.array([1,1,0,0])

count = 0
i = 0

x = np.random.shuffle(np.arange(4))

while count!=3:
    e = labels[i]-hardlim(w.dot(node[i])+b) #calculate e
    if e==0:
        count += 1
    else:
        count = 0
    w = w + e * node[i]
    b = b + e
    i = (i+1) % 4
    print("w1=%d, w2=%d, b=%d"%(w[0],w[1],b))