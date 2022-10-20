import numpy as np

"""
差分进化算法
"""
def mutation_fun(self):
        for i in range(self.size):
            r1 = r2 = r3 = 0
            while r1 == i or r2 == i or r3 == i or r2 == r1 or r3 == r1 or r3 == r2:
                r1 = random.randint(0, self.size - 1)  # 随机数范围为[0,size-1]的整数
                r2 = random.randint(0, self.size - 1)
                r3 = random.randint(0, self.size - 1)
            mutation = self.get_kth_unit(r1).get_pos() + \
                       self.F * (self.get_kth_unit(r2).get_pos() - self.get_kth_unit(r3).get_pos())
            for j in range(self.dim):
                #  判断变异后的值是否满足边界条件，不满足需重新生成
                if self.x_min <= mutation[j] <= self.x_max:
                    self.get_kth_unit(i).set_mutation(j, mutation[j])
                else:
                    rand_value = self.x_min + random.random()*(self.x_max - self.x_min)
                    self.get_kth_unit(i).set_mutation(j, rand_value)

    # 交叉
#    def crossover(self):
#        for unit in self.unit_list:
#            for j in range(self.dim):
#                rand_j = random.randint(0, self.dim - 1)
#                rand_float = random.random()
#                if rand_float <= self.CR or rand_j == j:
#                    unit.set_crossover(j, unit.get_mutation()[j])
#                else:
#                    unit.set_crossover(j, unit.get_pos()[j])
#
#    # 选择
#    def selection(self):
#        for unit in self.unit_list:
#            new_fitness_value = fit_fun(unit.get_crossover())
#            if new_fitness_value < unit.get_fitness_value():
#                unit.set_fitness_value(new_fitness_value)
#                for i in range(self.dim):
#                    unit.set_pos(i, unit.get_crossover()[i])
#            if new_fitness_value < self.get_bestFitnessValue():
#                self.set_bestFitnessValue(new_fitness_value)
#                for j in range(self.dim):
#                    self.set_bestPosition(j, unit.get_crossover()[j])
#
#    def update(self):
#        for i in range(self.iter_num):
#            self.mutation_fun()
#            self.crossover()
#            self.selection()
#            self.fitness_val_list.append(self.get_bestFitnessValue())
#        return self.fitness_val_list, self.get_bestPosition()





MAX_GENERATION = 100  # 最大迭代次数
GENERATION = 1  # 当前代数
BOUND_X = [-2, 2]  # 上下界
L = 6  # 种子的维度
N = 100  # 种群规模
SIGMA = 0.5  # 缩放因子
CR = 0.3  # 交叉概率
BEST_X = 0
FITS = np.zeros(N)
FITS_U = np.zeros(N)
V = np.zeros((N, L))
U = np.zeros((N, L))
mutationStrategy=1;#变异策略
crossStrategy=1;#交叉策略
E = 1e-15
T = 30
G = np.zeros(T)





BEST_X=pd.read_csv(r'BEST_X.csv')
data=np.array(BEST_X)
X=data[:,1:]

data_=pd.read_csv(r'LGBM_HXS_moxing.csv')
data1=np.array(data_)
BEST_X=data1[:,1:]
V=mutation_fun(X,BEST_X,SIGMA,mutationStrategy);
U=crossover(X,V,CR,crossStrategy);
