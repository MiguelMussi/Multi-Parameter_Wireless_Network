from typing import List, Tuple, Dict
import random
import numpy as np
import copy
import pickle
import time

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
	val = np.array(values)
	val = val.reshape(1,-1)
	R1,R2,R3 = model.predict(val)[0]
	return calc_score(R1,R2,R3)

class State:
	def __init__(self,case:int,parameters:List[List[int]]):
		self.case: int = case
		self.indices: List[int] = []
		self.values: List[int] = []
		self.parameters: List[List[int]] = parameters
		self.next_state_ant1:int
		self.next_state_ant2:int
		self.next_state_old_idx1:int
		self.next_state_old_idx2:int

	def gen_inicial(self):
		self.indices = [0]*56
		for ant in range(56):
			for j in range(4):
				self.values.append(self.parameters[ant*20 + self.indices[ant]][j])

	def gen_random(self):
		self.indices = [random.randint(0,19) for _ in range(56)]
		self.values = []
		for ant in range(56):
			for j in range(4):
				self.values.append(self.parameters[ant*20 + self.indices[ant]][j])

	def upd_all_values(self):
		for ant in range(56):
			self.upd_values(ant)


	def upd_values(self,antenna:int):
		self.values[antenna*4:antenna*4+4] = self.parameters[antenna*20 + self.indices[antenna]][:]


	def next_state(self):
		self.next_state_ant1 = np.random.randint(56)
		self.next_state_old_idx1 = self.indices[self.next_state_ant1]
		self.indices[self.next_state_ant1] = np.random.randint(20)

		while self.next_state_old_idx1==self.indices[self.next_state_ant1]:
			self.indices[self.next_state_ant1] = np.random.randint(20)

		self.upd_values(self.next_state_ant1)
		if np.random.randint(2):
			self.next_state_ant2 = np.random.randint(56)
			while self.next_state_ant2==self.next_state_ant1:
				self.next_state_ant2 = np.random.randint(56)

			self.next_state_old_idx2 = self.indices[self.next_state_ant2]
			self.indices[self.next_state_ant2] = np.random.randint(20)
			while self.next_state_old_idx2==self.indices[self.next_state_ant2]:
				self.indices[self.next_state_ant2] = np.random.randint(20)

			self.upd_values(self.next_state_ant2)
		else:
			self.next_state_ant2 = self.next_state_ant1

		return self.calc_state_score()


	def undo_next_state(self):
		self.indices[self.next_state_ant1] = self.next_state_old_idx1
		self.upd_values(self.next_state_ant1)
		if self.next_state_ant2!=self.next_state_ant1:
			self.indices[self.next_state_ant2] =self.next_state_old_idx2
			self.upd_values(self.next_state_ant2)


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
	with open("NN/trained_nn","rb") as f:
		model = pickle.load(f)
	return model

def load_patterns():
	patterns=[]
	filename = "attachment_info/patterns.csv"
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
		filename = "attachment_info/CASE_" + str(case) + ".csv"
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
			iters = 10000
			scores = set()
			while True:
				state = State(caso,case_parameters[caso])
				state.gen_random()
				score = state.calc_state_score()
				score,best_indices = HC(state,score,iters)
				if score in scores:
					continue
				best_indices = tuple(best_indices)
			#	if best_indices in indices_usados[caso]: continue
				scores.add(score)
				print("caso:",caso," len:",len(scores)," score:",score)
			#	indices_usados[caso].add(best_indices)
				f.write('[')
				f.write(','.join(map(str,best_indices)))
				if caso!=5 or len(scores)<50:
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


