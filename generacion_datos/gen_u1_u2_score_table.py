import os
import numpy as np
from sklearn.linear_model  import LinearRegression as LR

ANTENNAS = 56
RESULTS = 6


def parse_one_file(content):
    lines = content.split('\n')[3:-1]
    x1 = []
    x2 = []
    y = []
    i = 0
    while (True):
        ilinea = i*(ANTENNAS + RESULTS + 1) + 1
        if ilinea >= len(lines): break
        R1 = lines[ilinea + ANTENNAS][24:]
        assert(R1[:2] == "R1")
        R1 = float(R1[3:-1])
        R2 = lines[ilinea + ANTENNAS+1][24:]
        assert(R2[:2] == "R2")
        R2 = float(R2[3:-1])
        R3 = lines[ilinea + ANTENNAS+2][24:]
        assert(R3[:2] == "R3")
        R3 = float(R3[3:-1])
        u1 = 100 - max(R1,R2,R3)
        u2 = 100 - (R1+R2+R3)/3
        score = lines[ilinea + ANTENNAS + 5][24:]
        assert(score[:5] == "Score")
        score = float(score[6:])
        score = score * 110/1000.
        x1.append(u1)
        x2.append(u2)
        y.append(score)
        i += 1
    return x1,x2,y

def read_all_files(directory):
    x1 = []
    x2 = []
    y = []
    for filename in os.listdir(directory):
        print(filename)
        if os.path.isfile(os.path.join(directory, filename)):
            with open(os.path.join(directory, filename), 'r') as f:
                [x1_i,x2_i,y_i] = parse_one_file(f.read())
                x1 += x1_i
                x2 += x2_i
                y += y_i
            
    return x1,x2,y

x1,x2,y = read_all_files("submitions")

x1 = x1[:10]
x2 = x2[:10]
y = y[:10]
X = np.array(list(zip(x1,x2)))
Y = np.array(y).reshape(-1,1)

reg = LR().fit(X,Y)
u = reg.coef_.tolist()[0]
u1= 100 - 10./ u[0]
u2= 100 - 1./u[1]
print(u1,u2)



