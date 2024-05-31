from typing import List, Tuple, Dict
import random
import numpy as np
import pickle
import time

N_INPUT_x_ANTENNA = 4
def temp_schedule_lineal(t0,tf,pje):
    t = t0 + (tf-t0)*pje
    return t

def temp_schedule_exp(t0,tf,pje):
    return t0 * pow(tf/t0,pje)

def calc_score(R1:float,R2:float,R3:float):
    u1=100.-max(R1,R2,R3)
    u2=100.-(R1+R2+R3)/3.
    score1 = (u1 - u10) * invu10
    score2 = (u2 - u20) * invu20
    score = (score1 + score2) * score_scale
    return score

def get_prediction(values)->float:
    val  = np.array(values)
    val = val.reshape(1,-1)
    R1,R2,R3 = model.predict(val)[0]
    return calc_score(R1,R2,R3)

class State:
    def __init__(self,case:int,parameters:List[List[int]]):
        self.case: int = case
        self.indices: List[int] = []
        self.values: List[int] = []
        self.parameters: List[List[int]] = parameters
        self.next_state_ants:List[int] = [0,0,0]
        self.next_state_old_idxs:List[int] = [0,0,0]
        self.next_tipo:int
        self.sitios = [[0,1,2],[3,4,5],[6,7,8],[9,10],[11,12,13],[14,15,16],[17,18],[19,20,21],[22,23,24],[25,26,27],[28,29],[30,31,32],[33,34,35],[36,37,38],[39,40,41],[42,43],[44,45,46],[47,48,49],[53,54,55]]

    def gen_inicial(self):
        self.indices = [0]*56
        for ant in range(56):
            for j in range(N_INPUT_x_ANTENNA):
                self.values.append(self.parameters[ant*20 + self.indices[ant]][j])

    def gen_best(self):
        indices_caso = [[7,15,6,19,11,12,5,11,11,1,5,12,18,18,13,6,17,8,19,7,17,5,14,8,1,13,13,6,19,19,15,16,11,6,2,4,3,10,16,8,14,2,10,10,6,7,10,2,3,16,19,6,17,2,6,10],
                                  [1,16,12,16,14,11,14,8,6,2,12,18,16,3,14,10,19,15,17,3,6,3,1,14,19,13,3,1,1,19,5,5,8,12,4,0,13,4,15,8,12,11,8,6,6,1,19,15,12,17,13,8,1,17,12,3],
                                   [5,14,14,10,1,10,4,19,6,8,9,9,3,15,10,2,12,0,6,16,2,16,13,1,19,2,11,16,1,6,13,8,13,18,8,0,1,1,12,0,10,4,3,12,12,9,11,10,3,7,1,11,7,8,1,7],
                            [12,0,1,16,10,7,15,2,7,3,0,12,10,14,14,10,14,12,14,8,4,9,10,16,8,15,1,13,14,10,17,16,4,3,16,0,14,8,2,4,18,4,7,5,9,4,9,5,3,3,6,14,14,2,15,14],
                            [16,11,1,12,9,0,15,16,7,17,10,2,11,14,12,4,0,3,10,17,19,12,12,10,5,1,15,13,9,6,19,5,11,13,12,12,5,13,15,9,9,14,19,8,3,8,3,8,7,18,5,8,14,18,12,14],
                            [14,12,19,10,13,5,3,11,1,2,7,6,5,4,10,9,0,9,3,2,9,0,1,6,8,2,6,16,7,2,2,14,9,4,9,0,17,4,13,9,14,7,17,11,13,5,12,14,7,16,8,5,11,14,8,13]]


        self.indices = indices_caso[self.case]
        self.values = []
        for ant in range(56):
            for j in range(N_INPUT_x_ANTENNA):
                self.values.append(self.parameters[ant * 20 + self.indices[ant]][j])
        print(self.values)

    def gen_subrandom(self):
        self.indices = [random.randint(0,19) for _ in range(56)]
        self.indices[6] = 8
        self.indices[7] = 14 # 10
        self.indices[8] =  18 # 17
        self.indices[9] =  12
        self.indices[10] = 2
        self.indices[11] =  11
        self.indices[12] =  17 # 9
        self.indices[13] =  13
        self.indices[14] =  8 #0
        self.values = []
        for ant in range(56):
            for j in range(N_INPUT_x_ANTENNA):
                self.values.append(self.parameters[ant*20 + self.indices[ant]][j])

    def gen_random(self):
        self.indices = [random.randint(0,19) for _ in range(56)]
        self.values = []
        for ant in range(56):
            for j in range(N_INPUT_x_ANTENNA):
                self.values.append(self.parameters[ant*20 + self.indices[ant]][j])

    def upd_all_values(self):
        for ant in range(56):
            self.upd_values(ant)


    def upd_values(self,antenna:int):
        self.values[antenna*N_INPUT_x_ANTENNA:antenna*N_INPUT_x_ANTENNA+N_INPUT_x_ANTENNA] = self.parameters[antenna*20 + self.indices[antenna]][:]


    def next_state(self):
        self.next_tipo = np.random.randint(3)

        if self.next_tipo==2:
            sitio = np.random.randint(19)
            self.next_tipo = len(self.sitios[sitio])-1
            for i in range(self.next_tipo+1):
                ant = self.sitios[sitio][i]
                self.next_state_ants[i] = ant
                self.next_state_old_idxs[i] = self.indices[ant]
                self.indices[ant] = np.random.randint(20)
                self.upd_values(ant)
        else:
            self.next_state_ants[0] = np.random.randint(56)
            self.next_state_old_idxs[0] = self.indices[self.next_state_ants[0]]
            self.indices[self.next_state_ants[0]] = np.random.randint(20)
            while self.next_state_old_idxs[0]==self.indices[self.next_state_ants[0]]:
                self.indices[self.next_state_ants[0]] = np.random.randint(20)
            self.upd_values(self.next_state_ants[0])

            if self.next_tipo == 1:
                self.next_state_ants[1] = np.random.randint(56)
                while self.next_state_ants[1]==self.next_state_ants[0]:
                    self.next_state_ants[1] = np.random.randint(56)
                self.next_state_old_idxs[1] = self.indices[self.next_state_ants[1]]
                self.indices[self.next_state_ants[1]] = np.random.randint(20)
                while self.next_state_old_idxs[1]==self.indices[self.next_state_ants[1]]:
                    self.indices[self.next_state_ants[1]] = np.random.randint(20)
                self.upd_values(self.next_state_ants[1])


        return self.calc_state_score()


    def undo_next_state(self):
        for i in range(self.next_tipo+1):
            self.indices[self.next_state_ants[i]] = self.next_state_old_idxs[i]
            self.upd_values(self.next_state_ants[i])


    def calc_state_score(self):
        return get_prediction(self.values)

def sa(state:State,score:float,iters:int):
    best_score = score
    best_indices = state.indices.copy()
    t0 = 10
    tf = .1
    loop_iters = 20
    iters //= loop_iters
    for it in range(iters):
        if divmod(it,4095)[1] == 0:
            print(it,score)
        temp = temp_schedule_exp(t0,tf,it/iters)
        for _ in range(loop_iters):
            new_score = state.next_state()
            if new_score<score and (score - new_score>20 or np.exp((new_score-score)/temp)) < np.random.uniform():
                state.undo_next_state()
                continue
            score = new_score
            if score > best_score:
                best_score=score
                best_indices = state.indices.copy()

    return best_score,best_indices

def HC(state:State,score:float,iters:int):
    for it in range(iters):
        new_score = state.next_state()
        if new_score<score:
            state.undo_next_state()
            continue
        score=new_score

    indices = state.indices.copy()
    return score,indices

def HC_less(state:State,score:float,iters:int):
    for it in range(iters):
        new_score = state.next_state()
        if new_score>score:
            state.undo_next_state()
            continue
        score=new_score

    indices = state.indices.copy()
    return score,indices

def LAHC(state:State,score:float,iters:int):
    MEMORY_SZ = 20
    best_score = score
    best_indices = state.indices.copy()
    score_memory = np.full(MEMORY_SZ,best_score)
    idx = 0
    for it in range(iters):
        alt_score = state.next_state()
        if alt_score>=score or alt_score>=score_memory[idx]:
            score = alt_score
            if score > best_score:
                best_score = score
                best_indices = state.indices.copy()
        else:
            state.undo_next_state()

        score_memory[idx] = score
        idx += 1
        if idx==MEMORY_SZ: idx=0
    return score,best_indices

def load_predictor():
    with open("../NN/meta_model.pkl","rb") as f:
        model = pickle.load(f)
    return model

def load_patterns():
    patterns=[]
    filename = "../attachment_info/patterns.csv"
    with open(filename,"r") as f:
        head = f.readline().strip()
        for p in range(17):
            este = f.readline().strip().split()[1:]
            este = [int(i) for i in este]
            patterns.append(este)
    return patterns

def load_cases():
    case_parameters:List[List[List[int]]]=[]
    for case in range(6):
        filename = "../attachment_info/CASE_" + str(case) + ".csv"
        with open(filename,"r") as f:
            head = f.readline().strip()
            este = [f.readline().strip().split(",")[2:] for _ in range(20 * 56)]
            este = [[int(i) for i in j] for j in este]
            for e in este:
                pat = PATTERNS[e[-1]]
                e[-1] = pat[0]
                e.append(pat[1])

            case_parameters.append(este)
    return case_parameters

def load_indices_usados():
    used=[]
    for caso in range(6):
        used.append(set())
        filename = "indices_usados_" + str(caso) + ".txt"
        try:
            with open(filename,"r") as f:
                for line in f:
                    integers = tuple(int(x) for x in line.strip().split())
                    used[caso].add(integers)
        except Exception:
            pass


    return used

PATTERNS = load_patterns()
case_parameters = load_cases()
model = load_predictor()
u10 = 77.368574063
u20 = 83.907946914
invu10 = 10./(100. - u10)
invu20 = 1./(100 - u20)
score_scale = 100./11.

#indices_usados = load_indices_usados()
random.seed(time.time())


for ran in range(10):
    filename = "nuevos_indices_" + str(ran) + ".txt"
    with open(filename,"w") as f:
        f.write("[")
        for caso in range(6):
            iters = 5000 if caso==0 else 100
            scores = set()
            for t in range(50):
                state = State(caso,case_parameters[caso])
                state.gen_best()
                score = state.calc_state_score()
                if t:
                    score,best_indices = HC(state,score,iters)
                    if score in scores:
                        t -= 1
                        continue
                else:
                    best_indices = state.indices.copy()
                #	if best_indices in indices_usados[caso]: continue
                scores.add(score)
                print("caso:",caso," len:",len(scores)," score:",score)
                #	indices_usados[caso].add(best_indices)
                f.write('[')
                f.write(','.join(map(str,best_indices)))
                if caso!=5 or t+1<50:
                    f.write("],\n")
                else:
                    f.write("]")
                if len(scores)==50: break
            f.flush()
        f.write(']')


#
# for caso in range(6):
# 	filename = "indices_usados_" + str(caso) + ".txt"
# 	with open(filename,"w") as f:
# 		for line in indices_usados[caso]:
# 			f.write(' '.join(map(str,line)))
# 			f.write('\n')


