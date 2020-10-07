import pandas as pd
import numpy as np
from time import time
import random


def initialization(Npop, themes):
    pop = []
    for i in range(Npop):
        p = list(np.random.permutation(len(themes)))
        while p in pop:
            p = list(np.random.permutation(len(themes)))
        pop.append(p)
    
    return pop

def initialization_greedy(Npop, themes, train_data, W):
    t0 = time()
    pop = []
    for i in range(Npop):
        used_themes = {}
        p = len(themes) * [0]
        rand_order = list(np.random.permutation(len(themes)))
        for j in rand_order:
            best_thm = -1
            best_obj = 0
            for thm in range(len(themes)):
                if thm not in used_themes:
                    p[j] = thm
                    
                    prediction_scores = []
                    temp_pred_scores = []
                    for ind, doc_wth in enumerate(W[train_data.index]): # only train data
                        temp_pred_score = []
                        for theme in train_data.iloc[ind]['theme']:
                            real_theme_id = themes.index(theme)
                            if real_theme_id == j:
                                theme_id = p[real_theme_id]
                                temp_pred_score.append(np.log(len(themes))-np.log(np.argwhere(doc_wth.argsort()[::-1]==theme_id)[0][0] + 1))
                        temp_pred_scores.append(temp_pred_score)
                    prediction_scores = np.array([sum(tps) for tps in temp_pred_scores])
                    curr_obj = prediction_scores.sum()
                    
                    if curr_obj > best_obj:
                        best_obj = curr_obj
                        best_thm = thm
            if best_thm == -1:
                random_choise_list = []
                for rcl in range(len(themes)):
                    if rcl not in used_themes:
                        random_choise_list.append(rcl)
                best_thm = np.random.choice(random_choise_list)
            used_themes[best_thm] = 1
            p[j] = best_thm
        pop.append(p)
    print("initialization complete", time()-t0)
    
    return pop

# def initialization_greedy(Npop, themes, train_data, W):
#     pop = []
#     for i in range(Npop):
#         used_themes = {}
#         p = len(themes) * [0]
#         rand_order = list(np.random.permutation(len(themes)))
#         for j in rand_order:
#             best_thm = -1
#             best_obj = 0
#             for thm in range(len(themes)):
#                 if thm not in used_themes:
#                     p[j] = thm
#                     curr_obj = calculateObj(p, train_data, W, themes, log_scoring = True)
#                     if curr_obj > best_obj:
#                         best_obj = curr_obj
#                         best_thm = thm
#             used_themes[best_thm] = 1
#             p[j] = best_thm
#         pop.append(p)
    
#     return pop

def calculateObj(sol, train_data, W, themes):
    prediction_scores = []
    temp_pred_scores = []
    for ind, doc_wth in enumerate(W[train_data.index]):
        temp_pred_score = []
        for theme in train_data.iloc[ind]['theme']:
            real_theme_id = themes.index(theme)
            theme_id = sol[real_theme_id]
            temp_pred_score.append(np.log(len(themes))-np.log(np.argwhere(doc_wth.argsort()[::-1]==theme_id)[0][0] + 1))
        temp_pred_scores.append(temp_pred_score)
    prediction_scores = np.array([sum(tps) for tps in temp_pred_scores])

    return prediction_scores.sum()


# def calculateObj(sol, train_data, W, themes, log_scoring = True):
#     prediction_scores = []
#     temp_pred_scores = []
#     for ind, doc_wth in enumerate(W):
#         temp_pred_score = []
#         for theme in train_data.iloc[ind]['theme']:
#             real_theme_id = themes.index(theme)
#             theme_id = sol[real_theme_id]
# #             if np.argwhere(doc_wth.argsort()==theme_id)[0][0] == 0:
# #                 temp_pred_score.append(0)
# #             else:
#             if log_scoring:
#                 temp_pred_score.append(np.log(np.argwhere(doc_wth.argsort()==theme_id)[0][0] + 1))
#             else:
#                 temp_pred_score.append(np.argwhere(doc_wth.argsort()==theme_id)[0][0] + 1)
#         temp_pred_scores.append(temp_pred_score)
#     prediction_scores = np.array([sum(tps) for tps in temp_pred_scores])

#     return prediction_scores.sum()
        

def selection(pop, popObjValues):
    popObj = []
    for i in range(len(pop)):
        popObj.append([popObjValues[i], i])
    
    popObj.sort()
    
    distr = []
    distrInd = []
    
    for i in range(len(pop)):
        distrInd.append(popObj[i][1])
        prob = (2*(i+1)) / (len(pop) * (len(pop)+1))
        distr.append(prob)
    
    parents = []
    for i in range(len(pop)):
        parents.append(list(np.random.choice(distrInd, 2, p=distr)))
    
    return parents

def crossover(parents, themes):
    pos = list(np.random.permutation(np.arange(len(themes)-1)+1)[:2])
    
    if pos[0] > pos[1]:
        t = pos[0]
        pos[0] = pos[1]
        pos[1] = t
    
    child = list(parents[0])
    
    for i in range(pos[0], pos[1]):
        child[i] = -1
    
    p = -1
    for i in range(pos[0], pos[1]):
        while True:
            p = p + 1
            if parents[1][p] not in child:
                child[i] = parents[1][p]
                break
    
    return child


def mutation(sol, themes):
    pos = list(np.random.permutation(np.arange(len(themes)))[:2])
    
    if pos[0] > pos[1]:
        t = pos[0]
        pos[0] = pos[1]
        pos[1] = t
    
    remJob = sol[pos[1]]
    
    for i in range(pos[1], pos[0], -1):
        sol[i] = sol[i-1]
        
    sol[pos[0]] = remJob
    
    return sol
        
def elitistUpdate(oldPop, newPop, oldPopObjValues):
    bestSolInd = 0
    bestSol = oldPopObjValues[0]
    
    for i in range(1, len(oldPop)):
        tempObj = oldPopObjValues[i]
        if tempObj > bestSol:
            bestSol = tempObj
            bestSolInd = i
            
    rndInd = random.randint(0,len(newPop)-1)
    
    newPop[rndInd] = oldPop[bestSolInd]
    
    return newPop.copy()

# Returns best solution's index number, best solution's objective value and average objective value of the given population.
def findBestSolution(pop, train_data, W, themes):
    popObjValues = []
    for p in pop:
        popObjValues.append(calculateObj(p, train_data, W, themes))
        
    bestObj = popObjValues[0]
    avgObj = bestObj
    bestInd = 0
    for i in range(1, len(pop)):
        tObj = popObjValues[i]
        avgObj = avgObj + tObj
        if tObj > bestObj:
            bestObj = tObj
            bestInd = i
            
    return bestInd, bestObj, avgObj/len(pop), popObjValues



# Function to split train-test dataset for multilabeled datasets

# https://vict0rs.ch/2018/05/24/sample-multilabel-dataset/
def stratify(data, classes, ratios, one_hot=False):
    """Stratifying procedure.

    data is a list of lists: a list of labels, for each sample.
        Each sample's labels should be ints, if they are one-hot encoded, use one_hot=True
    
    classes is the list of classes each label can take

    ratios is a list, summing to 1, of how the dataset should be split

    """
    # one-hot decoding
    if one_hot:
        temp = [[] for _ in range(len(data))]
        indexes, values = np.where(np.array(data).astype(int) == 1)
        for k, v in zip(indexes, values):
            temp[k].append(v)
        data = temp

    # Organize data per label: for each label l, per_label_data[l] contains the list of samples
    # in data which have this label
    per_label_data = {c: set() for c in classes}
    for i, d in enumerate(data):
        for l in d:
            per_label_data[l].add(i)

    # number of samples
    size = len(data)

    # In order not to compute lengths each time, they are tracked here.
    subset_sizes = [r * size for r in ratios]
    target_subset_sizes = np.copy(subset_sizes)
    per_label_subset_sizes = {
        c: [r * len(per_label_data[c]) for r in ratios]
        for c in classes
    }

    # For each subset we want, the set of sample-ids which should end up in it
    stratified_data_ids = [set() for _ in range(len(ratios))]

    # For each sample in the data set
    while size > 0:
        # Compute |Di|
        lengths = {
            l: len(label_data)
            for l, label_data in per_label_data.items()
        }
        try:
            # Find label of smallest |Di|
            label = min(
                {k: v for k, v in lengths.items() if v > 0}, key=lengths.get
            )
        except ValueError:
            # If the dictionary in `min` is empty we get a Value Error. 
            # This can happen if there are unlabeled samples.
            # In this case, `size` would be > 0 but only samples without label would remain.
            # "No label" could be a class in itself: it's up to you to format your data accordingly.
            break
        current_length = lengths[label]

        # For each sample with label `label`
        while per_label_data[label]:
            # Select such a sample
            current_id = per_label_data[label].pop()

            subset_sizes_for_label = per_label_subset_sizes[label]
            # Find argmax clj i.e. subset in greatest need of the current label
            largest_subsets = np.argwhere(
                subset_sizes_for_label == np.amax(subset_sizes_for_label)
            ).flatten()

            if len(largest_subsets) == 1:
                subset = largest_subsets[0]
            # If there is more than one such subset, find the one in greatest need
            # of any label
            else:
                largest_subsets = np.argwhere(
                    subset_sizes == np.amax(subset_sizes)
                ).flatten()
                if len(largest_subsets) == 1:
                    subset = largest_subsets[0]
                else:
                    # If there is more than one such subset, choose at random
                    subset = np.random.choice(largest_subsets)

            # Store the sample's id in the selected subset
            stratified_data_ids[subset].add(current_id)

            # There is one fewer sample to distribute
            size -= 1
            # The selected subset needs one fewer sample
            subset_sizes[subset] -= 1

            # In the selected subset, there is one more example for each label
            # the current sample has
            for l in data[current_id]:
                per_label_subset_sizes[l][subset] -= 1
            
            # Remove the sample from the dataset, meaning from all per_label dataset created
            for l, label_data in per_label_data.items():
                if current_id in label_data:
                    label_data.remove(current_id)

    # Create the stratified dataset as a list of subsets, each containing the orginal labels
    stratified_data_ids = [sorted(strat) for strat in stratified_data_ids]
    stratified_data = [
        [data[i] for i in strat] for strat in stratified_data_ids
    ]

    # Return both the stratified indexes, to be used to sample the `features` associated with your labels
    # And the stratified labels dataset
    return stratified_data_ids, stratified_data

# def split_train_test(data, themes, train_test_split):
#     '''
#     Function to split train-test dataset for multilabeled datasets.
#     First shuffles the dataset according to random_staten then splits it.
#     For more info visit: https://vict0rs.ch/2018/05/24/sample-multilabel-dataset/

#     Returns train and test data.
#     '''
#     # Shuffle dataset to get randomized split
#     data_shuffled = data.sample(frac=1).sort_values('theme').reset_index(drop=True).copy(deep=True)

#     #self.themes = sorted(list(set(data_shuffled['theme'].sum())))
#     themes_ids = [i for i in range(len(themes))]

#     doc_theme_ids = []
#     for doc_themes in list(data_shuffled['theme']):
#         temp = []
#         for theme in doc_themes:
#             temp.append(themes.index(theme))
#         doc_theme_ids.append(temp)

#     stratified_data_ids, stratified_data = stratify(data=doc_theme_ids, classes=themes_ids, ratios=train_test_split, one_hot=False)
    
#     train_data = data_shuffled.copy(deep=True)
#     test_data = data_shuffled.iloc[stratified_data_ids[1]].copy(deep=True)

#     return train_data, test_data


def calculateTestScore(sol, test_data, W, themes):
    all_prediction_scores = []
    for ind, doc_wth in enumerate(W[test_data.index]):
        temp_pred_score = []
        for theme in test_data.iloc[ind]['theme']:
            real_theme_id = themes.index(theme)
            theme_id = sol[real_theme_id]
            temp_pred_score.append(np.log(len(themes))-np.log(np.argwhere(doc_wth.argsort()[::-1]==theme_id)[0][0] + 1))
        all_prediction_scores.append(temp_pred_score)
    prediction_scores = [sum(aps) for aps in all_prediction_scores]
    
    prediction_scores = np.array(prediction_scores)
    all_prediction_scores = np.array(all_prediction_scores)

    return prediction_scores, all_prediction_scores


# def calculateTestScore(sol, test_data, W, themes, log_scoring = True):
#     all_prediction_scores = []
#     for ind, doc_wth in enumerate(W[test_data.index]):
#         temp_pred_score = []
#         for theme in test_data.iloc[ind]['theme']:
#             real_theme_id = themes.index(theme)
#             theme_id = sol[real_theme_id]
# #             if np.argwhere(doc_wth.argsort()==theme_id)[0][0] == 0:
# #                 temp_pred_score.append(0)
# #             else:
#             if log_scoring:
#                 temp_pred_score.append(np.log(np.argwhere(doc_wth.argsort()==theme_id)[0][0] + 1))
#             else:
#                 temp_pred_score.append(np.argwhere(doc_wth.argsort()==theme_id)[0][0] + 1)
        
        
#         all_prediction_scores.append(temp_pred_score)
#     prediction_scores = [sum(aps) for aps in all_prediction_scores]
    
#     prediction_scores = np.array(prediction_scores)
#     all_prediction_scores = np.array(all_prediction_scores)

#     return prediction_scores, all_prediction_scores


def run_ga(train_data, W, themes, Npop = 3, Pc = 1.0, Pm = 1.0, stopGeneration = 10, greedyInit = False):
    '''
    Runs GA with the given paramters and returns the solutions
    
    Parameters:
    -----------
    train_data : Train Data in DataFrame Format
    W : Doc-Topic Matrix
    themes : Themes list of Data
    Npop : Number of population
    Pc : Probability of crossover
    Pm : Probability of mutation
    stopGeneration : Stopping number for generation
    greedyInit: Initialize with greedy approach
    
    Returns:
    ---------
    population: All population
    population[bestSol]: Best individual
    bestSol: Index of the best individual
    bestObj: Objective value for the best individual
    
    '''

    # Start Timer
    t1 = time()

    # Creating the initial population
    if greedyInit:
        population = initialization_greedy(Npop, themes, train_data, W)
    else:
        population = initialization(Npop, themes)
        
    _, _, _, popObjValues = findBestSolution(population, train_data, W, themes)

    # Run the algorithm for 'stopGeneration' times generation
    for i in range(stopGeneration):
        # Selecting parents
        parents = selection(population, popObjValues)
        childs = []

        # Apply crossover
        for p in parents:
            r = random.random()
            if r < Pc:
                childs.append(crossover([population[p[0]], population[p[1]]].copy(), themes))
            else:
                if r < 0.5:
                    childs.append(population[p[0]].copy())
                else:
                    childs.append(population[p[1]].copy())

        # Apply mutation 
        for c in childs:
            r = random.random()
            if r < Pm:
                c = mutation(c, themes)

        # Update the population
        population = elitistUpdate(population, childs, popObjValues)

        bestSol, bestObj, avgObj, popObjValues = findBestSolution(population, train_data, W, themes)
        #print(bestSol, bestObj)
        
    return population, population[bestSol], bestSol, bestObj