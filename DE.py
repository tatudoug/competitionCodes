import random
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import lightgbm as lgb
import pandas as pd
import numpy as np

# --- EXAMPLE COST FUNCTIONS ---------------------------------------------------+

globvar = 0.1

def func1(x,a,b):
    # Sphere function, use any bounds, f(0,...,0)=0
    return 1/(sum([x[i] ** 2 for i in range(len(x))]))



def f_1(preds, train_data):
    labels = train_data.get_label()
    return 'error', f1_score(labels,np.round(preds-globvar)), True

def lgbm_tra(x,X,y):

# seleciona os parametros para treinamento
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'xentropy',#'binary',
        'num_leaves': int(x[1]),      # 31
        'learning_rate': x[2],   #0.01,
        'feature_fraction': x[3],#0.9,
        'bagging_fraction': x[4],#0.8,
        'bagging_freq': 1,
        'max_depth': int(x[5]),        #-1,
        'min_data_in_leaf': int(x[6]), #20,
        'lambda_l2': x[7],        # 0,
        'is_unbalance' : True,
        'max_bin' : int(x[9]),
        'verbose': -1
    }
    num_k = 5
    acc = 0
# treina a arvore
    kf = KFold(n_splits=num_k,random_state=21)

    global globvar    # Needed to modify global copy of globvar
    globvar = x[8]

    for train_index, test_index in kf.split(X):
       #print("TRAIN:", train_index, "TEST:", test_index)
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]

       lgb_train = lgb.Dataset(X_train, y_train)


       lgb_test = lgb.Dataset(X_test, y_test)
       #gbm = lgb.train(params, lgb_train, num_boost_round=int(x[0])  ) #x[0]

       gbm = lgb.train(params, lgb_train, num_boost_round=int(x[0]),valid_sets = [lgb_test],early_stopping_rounds=30,feval = f_1, verbose_eval= 0)     #x[0]

       y_pred = gbm.predict(X_test,num_iteration=gbm.best_iteration)
       acc += f1_score(y_test,np.round(y_pred-x[8]) ) # F1 score

    return (acc/num_k)


# --- FUNCTIONS ----------------------------------------------------------------+


def ensure_bounds(vec, bounds):
    vec_new = []
    # cycle through each variable in vector
    for i in range(len(vec)):

        # variable exceedes the minimum boundary
        if vec[i] < bounds[i][0]:
            vec_new.append(bounds[i][0])

        # variable exceedes the maximum boundary
        if vec[i] > bounds[i][1]:
            vec_new.append(bounds[i][1])

        # the variable is fine
        if bounds[i][0] <= vec[i] <= bounds[i][1]:
            vec_new.append(vec[i])

    return vec_new


# --- MAIN ---------------------------------------------------------------------+

def main(cost_func, bounds, popsize, mutate, recombination, maxiter,x_train,y_train):
    # --- INITIALIZE A POPULATION (step #1) ----------------+

    population = []
    for i in range(0, popsize):
        indv = []
        for j in range(len(bounds)):
            indv.append(random.uniform(bounds[j][0], bounds[j][1]))
        population.append(indv)

    print('First gen ...')
    scorePop = np.zeros(popsize)
    for num, ind in enumerate(population):
        scorePop[num] = cost_func(ind, x_train, y_train)
        print(scorePop[num])

    # --- SOLVE --------------------------------------------+

    # cycle through each generation (step #2)
    for i in range(1, maxiter + 1):
        print('GENERATION:', i)

        gen_scores = []  # score keeping

        # cycle through each individual in the population
        for j in range(0, popsize):

            # --- MUTATION (step #3.A) ---------------------+

            # select three random vector index positions [0, popsize), not including current vector (j)
            canidates = np.arange(0, popsize)
            canidates = np.delete(canidates, j)

            random_index = np.random.permutation(canidates)

            x_1 = population[random_index[0]]
            x_2 = population[random_index[1]]
            x_3 = population[random_index[2]]
            x_t = population[j]  # target individual

            # subtract x3 from x2, and create a new vector (x_diff)
            x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]

            # multiply x_diff by the mutation factor (F) and add to x_1
            v_donor = [x_1_i + mutate * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
            v_donor = ensure_bounds(v_donor, bounds)

            # --- RECOMBINATION (step #3.B) ----------------+

            v_trial = []
            for k in range(len(x_t)):
                crossover = random.random()
                if crossover <= recombination:
                    v_trial.append(v_donor[k])

                else:
                    v_trial.append(x_t[k])

            # --- GREEDY SELECTION (step #3.C) -------------+

            score_trial = cost_func(v_trial,x_train,y_train)
            score_target = scorePop[j]

            if score_trial > score_target:
                population[j] = v_trial
                scorePop[j] = score_trial
                #print(score_target, '   <', score_trial, v_trial)

            else:
                pass
                #print(score_target,'   >', score_target, x_t)


        # --- SCORE KEEPING --------------------------------+
        gen_avg = np.mean(scorePop)  # current generation avg. fitness
        gen_best = max(scorePop)  # fitness of best individual
        gen_sol = population[np.argmax(scorePop)]  # solution of best individual

        print('      > GENERATION AVERAGE:', gen_avg)
        print('      > GENERATION BEST:', gen_best)
        print('         > BEST SOLUTION:', gen_sol, '\n')

    return gen_sol


# --- CONSTANTS ----------------------------------------------------------------+

cost_func = func1  # Cost function
#bounds = [(-1, 1), (-1, 1)]  # Bounds [(x1_min, x1_max), (x2_min, x2_max),...]

bounds = [(200, 1000),(15,50), (0.001,0.1 ), (0.5, 1),(0.7, 1), (6, 30),(10, 30),(0,20),(-0.4,0.4),(32,128)]  # Bounds [(x1_min, x1_max), (x2_min, x2_max),...]

"""
        'num_leaves': int(x[1]),      # 31
        'learning_rate': x[2],   #0.01,
        'feature_fraction': x[3],#0.9,
        'bagging_fraction': x[4],#0.8,
        'max_depth': int(x[5]),        #-1,
        'min_data_in_leaf': int(x[6]), #20,
        'lambda_l2': x[7],        # 0,
"""

popsize = 10  # Population size, must be >= 4
mutate = 0.5  # Mutation factor [0,2]
recombination = 0.7  # Recombination rate [0,1]
maxiter = 2000  # Max number of generations (maxiter)

# --- RUN ----------------------------------------------------------------------+
#print("start")
#main(cost_func, bounds, popsize, mutate, recombination, maxiter,1,1)





# load or create your dataset
print('Load data...')
df_train = pd.read_csv("dados/train.csv")
#print(df_train.head())
df_test = pd.read_csv('dados/test.csv')
#print(df_test.head())

y_train = df_train['is_promoted'].values

#print(df_train.dtypes)
#print(df_train["department"].value_counts())
#print(df_train["region"].value_counts())
#print(df_train["education"].value_counts()) #ok
#print(df_train["gender"].value_counts()) #ok
#print(df_train["recruitment_channel"].value_counts()) #ok


cleanup_nums = {"gender":     {"m": 1, "f": 0},
                "recruitment_channel": {"other": 0, "sourcing": 0.5, "referred": 1 },
                "education": {"Below Secondary": 0, "Bachelor's": 0.5, "Master's & above": 1}
                }

df_train.replace(cleanup_nums, inplace=True)
df_test.replace(cleanup_nums, inplace=True)


#df_train = pd.get_dummies(df_train, columns=["recruitment_channel"], prefix=["rec"])
#df_test = pd.get_dummies(df_test, columns=["recruitment_channel"], prefix=["rec"])

#df_train["region_mod"] = df_train["region"].astype('category').cat.codes
#df_test["region_mod"] = df_test["region"].astype('category').cat.codes

df_train = pd.get_dummies(df_train, columns=["department"], prefix=["dep"])
df_test = pd.get_dummies(df_test, columns=["department"], prefix=["dep"])


X_train = df_train.drop(['region','is_promoted','employee_id'], axis=1).values
X_test = df_test.drop(['region','employee_id'], axis=1).values


print('Train model ...')

cost_func = lgbm_tra
main(cost_func, bounds, popsize, mutate, recombination, maxiter,X_train,y_train)