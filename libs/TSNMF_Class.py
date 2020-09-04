import pandas as pd
import numpy as np
from time import time
import random

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation as LDA

class TSNMF:
    ''' Theme Supervised Non-Negative Matrix Factorization (TSNMF)
    
    Parameters
    ----------
    data : Pandas DataFrame that includes id, theme and text as columns.
        id (string) is a unique identifier of the document.
        theme (list of string) is the list of themes for the document.
        text (string) is the text of the document as a string
        
    supervision : string, default 'supervised'.
        String must be in {'supervised', 'semi_supervised'}.
        supervised: Does not include test set to the TSNMF model while training.
        semi_supervised: Trains TSNMF model with test set without label info.
        
    separate_models : Bool, default True
        if True then generates separate TSNMF models for each theme,
        Otherwise train all the themes together.
        
    bCool_init : Bool, defaul False
        if True then uses bCool initialization method to initialize theme-term matrix.
        Otherwise initialize randomly.
        
    train_test_split : list of 2 percentage, default [0.7, 0.3]
        Numbers in the list have to sum up to 1.
        First percentage is for train, second one is for test.
    
    n_topics : int, default 3
        Number of topics that will be used for each theme.
        
    n_terms = int, default 10000
        Number of terms to be used in the dictionary
        
    background_for_theme : Bool, default True
        if True then include theme-document to the background
    
    background_scoring : Bool, default True
        if True use background theme for scoring (in W_test_high)
        
    beta_loss : string, default 'frobenius'
        String must be in {'frobenius', 'kullback-leibler'}.
        Beta divergence to be minimized, measuring the distance between X
        and the dot product WH.
        
    term_vectorizer : string, default 'tf'
        String must be in {'tf', 'tfidf'}.
        tf: term frequency.
        tfidf: term frequenct inverse document frequency
        
    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
        
        
     Attributes
    ----------
    themes : list of String
        Themes in alphabetic order
    train_data : list of String
        ids of train data
    train_data : list of String
        ids of test data
    '''
    
    
    def __init__(self, data = None, supervision = 'supervised', separate_models = True, bCool_init = False, train_test_split = [0.7, 0.3], n_topics = 3, n_terms = 10000,
                 background_for_theme = True, background_scoring = True, beta_loss = 'frobenius', term_vectorizer = 'tf', random_state = None):
        self.data = data
        self.supervision = supervision
        self.separate_models = separate_models
        self.bCool_init = bCool_init
        self.train_test_split = train_test_split
        self.n_topics = n_topics
        self.n_terms = n_terms
        self.background_for_theme = background_for_theme
        self.background_scoring = background_scoring
        self.beta_loss = beta_loss
        self.term_vectorizer = term_vectorizer
        self.random_state = random_state
        
        self.themes = sorted(list(set(data['theme'].sum())))
        np.random.seed(random_state)
        
        
    def split_train_test(self):
        '''
        Function to split train-test dataset for multilabeled datasets.
        First shuffles the dataset according to random_staten then splits it.
        For more info visit: https://vict0rs.ch/2018/05/24/sample-multilabel-dataset/
        
        Creates train and test data.
        Returns self.
        '''
        # Shuffle dataset to get randomized split
        data_shuffled = self.data.sample(frac=1).sort_values('theme').reset_index(drop=True).copy(deep=True)

        #self.themes = sorted(list(set(data_shuffled['theme'].sum())))
        themes_ids = [i for i in range(len(self.themes))]

        doc_theme_ids = []
        for doc_themes in list(data_shuffled['theme']):
            temp = []
            for theme in doc_themes:
                temp.append(self.themes.index(theme))
            doc_theme_ids.append(temp)

        stratified_data_ids, stratified_data = stratify(data=doc_theme_ids, classes=themes_ids, ratios=self.train_test_split, one_hot=False)
        
        
        if self.supervision == 'supervised':
            self.train_data = data_shuffled.iloc[stratified_data_ids[0]].reset_index(drop=True).copy(deep=True)
            self.test_data = data_shuffled.iloc[stratified_data_ids[1]].reset_index(drop=True).copy(deep=True)
        else:
            self.train_data = data_shuffled.copy(deep=True)
            self.train_data['labeled'] = 1
            self.train_data.loc[stratified_data_ids[1], 'labeled'] = 0
            self.test_data = data_shuffled.iloc[stratified_data_ids[1]].copy(deep=True)
        
        return self
    
    
    def split_train_test_forced(self):
        '''
        Ensures that both train and test dataset have documents with all the labels.
        First randomly separates minimum number of documents that have all the labels
        for both train and test dataset.
        Then applys the same procedure with split_train_test()
        Finally adds separated documents to the final train and test datasets.
        
        Creates train and test data.
        Returns self.
        '''
        # Shuffle dataset to get randomized split
        data_shuffled = self.data.sample(frac=1).reset_index(drop=True).copy(deep=True)

        theme_train_dict = {}
        theme_train_ids = []
        theme_test_dict = {}
        theme_test_ids = []

        theme_ctr = 0
        for ind, row in data_shuffled.iterrows():
            for theme in row['theme']:
                if theme not in theme_train_dict:
                    theme_train_dict[theme] = 1
                    theme_ctr += 1
                    theme_train_ids.append(ind)
            if theme_ctr >= 90:
                break
        theme_train_ids = list(set(theme_train_ids))
        train_ds = data_shuffled.loc[theme_train_ids].reset_index(drop=True).copy(deep=True)
        #print(theme_ctr)
        
        data_shuffled = data_shuffled.drop(theme_train_ids).reset_index(drop=True).copy(deep=True)

        theme_ctr = 0
        for ind, row in data_shuffled.iterrows():
            for theme in row['theme']:
                if theme not in theme_test_dict:
                    theme_test_dict[theme] = 1
                    theme_ctr += 1
                    theme_test_ids.append(ind)
            if theme_ctr >= 90:
                break
        theme_test_ids = list(set(theme_test_ids))
        test_ds = data_shuffled.loc[theme_test_ids].reset_index(drop=True).copy(deep=True)
        #print(theme_ctr)
        
        data_shuffled = data_shuffled.drop(theme_test_ids).sort_values('theme').reset_index(drop=True).copy(deep=True)

        
        #self.themes = sorted(list(set(data_shuffled['theme'].sum())))
        themes_ids = [i for i in range(len(self.themes))]

        doc_theme_ids = []
        for doc_themes in list(data_shuffled['theme']):
            temp = []
            for theme in doc_themes:
                temp.append(self.themes.index(theme))
            doc_theme_ids.append(temp)

        stratified_data_ids, stratified_data = stratify(data=doc_theme_ids, classes=themes_ids, ratios=self.train_test_split, one_hot=False)
        
        
        if self.supervision == 'supervised':
            self.train_data = data_shuffled.iloc[stratified_data_ids[0]].reset_index(drop=True).copy(deep=True)
            self.train_data = self.train_data.append(train_ds).sort_values('theme').reset_index(drop=True).copy(deep=True)
            
            self.test_data = data_shuffled.iloc[stratified_data_ids[1]].reset_index(drop=True).copy(deep=True)
            self.test_data = self.test_data.append(test_ds).sort_values('theme').reset_index(drop=True).copy(deep=True)
        else:
            self.train_data = data_shuffled.copy(deep=True)
            self.train_data['labeled'] = 1
            self.train_data.loc[stratified_data_ids[1], 'labeled'] = 0
            train_ds['labeled'] = 1
            test_ds['labeled'] = 0
            self.train_data = self.train_data.append([train_ds, test_ds]).sort_values('theme').reset_index(drop=True).copy(deep=True)
            
            self.test_data = self.train_data[self.train_data['labeled']==0].copy(deep=True)
            #self.test_data = data_shuffled.iloc[stratified_data_ids[1]].copy(deep=True)
        
        return self
    
    
    def create_train_test(self, train_indices, test_indices):
        '''
        Creates custom train and test datasets from the original data using row indices. 
        
        Returns self.
        '''
        if self.supervision == 'supervised':
            self.train_data = data.iloc[train_indices].reset_index(drop=True).copy(deep=True)
            self.test_data = data.iloc[test_indices].reset_index(drop=True).copy(deep=True)
        else:
            self.train_data = data.copy(deep=True)
            self.train_data['labeled'] = 1
            self.train_data.loc[test_indices, 'labeled'] = 0
            self.test_data = data.iloc[test_indices].copy(deep=True)
        
        return self
    
    def term_vectorization(self, corpus):
        if self.term_vectorizer == 'tf':
            #print("Extracting tf features for NMF...", end = ' ')
            tf_vectorizer = CountVectorizer(min_df=1, ngram_range=(1,3), max_features=self.n_terms)
        else:
            #print("Extracting tf-idf features for NMF...", end = ' ')
            tf_vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1,3), max_features=self.n_terms)

        tf = tf_vectorizer.fit_transform(corpus)
        n_features = tf.shape[1]
        
        return tf, tf_vectorizer, n_features
        
    
    def generate_W(self):
        n_docs = len(self.train_data)
        
        supervision = np.zeros((n_docs,len(self.themes)*self.n_topics))
        if self.supervision == 'supervised':
            for ind, row in self.train_data.iterrows():
                for th in row['theme']:
                    supervision[ind, self.n_topics*self.themes.index(th):self.n_topics*(self.themes.index(th)+1)] = 1
        else:
            for ind, row in self.train_data.iterrows():
                if row['labeled'] == 0:
                    supervision[ind, :] = 1
                else:
                    for th in row['theme']:
                        supervision[ind, self.n_topics*self.themes.index(th):self.n_topics*(self.themes.index(th)+1)] = 1
                
                
        
        if self.separate_models == True:
            W_list = []

            for i in range(len(self.themes)):
                W = np.random.random((n_docs,self.n_topics+1))
                W[:, :-1] *= supervision[:, i*self.n_topics:(i+1)*self.n_topics]
                if not self.background_for_theme:
                    if self.supervision == 'supervised':
                        W[:, self.n_topics:] *= (1-supervision[:, i*self.n_topics])[:, None]
                    else:
                        labeled_nontheme_docs = np.array(1-supervision[:, i*self.n_topics], dtype=np.int64)
                        unlabeled_docs = np.array(1-self.train_data['labeled'])
                        W[:, self.n_topics:] *= (labeled_nontheme_docs | unlabeled_docs)[:, None]
                W_list.append(W)
            return W_list
        else:
            W = np.random.random((n_docs,len(self.themes)*self.n_topics+1))
            W[:, :-1] *= supervision 
            return W
        
    
    def bCool_initialization_H(self, tf):
        '''
        Initialize H (Theme-Term) matrix.
        
        Parameters:
        -----------
        tf : term frequency matrix.
        
        Returns:
        -----------
        H/H_list : Theme-Term Matrix
        '''
        
        X = tf.toarray() # Convert tf to matrix
        X_sum = list(np.sum(X>0, axis=1)) # find count of nonempty rows for each document (density of a doc)
        X_sum = [(x, i) for i, x in enumerate(X_sum)] # Put document id next to scores
        
        # X_sum_list holds X_sum values but separated by themes. Every index of X_sum_list is for a different theme.
        # So if a document has more than one theme, then it will appear more than one index of X_sum_list
        X_sum_list = [[] for i in range(len(self.themes))]
        theme_ids_list = list(self.train_data['theme'].apply(lambda t_list: [self.themes.index(t) for t in t_list])) # Get theme indices for each document
        for doc_order, theme_ids in enumerate(theme_ids_list):
            if self.supervision == 'semi_supervised' and self.train_data.loc[doc_order, 'labeled'] == 0: # If it is labeled as 0 do not take (for semi_supervised)
                continue
            for theme_id in theme_ids:
                X_sum_list[theme_id].append(X_sum[doc_order])
        
        # Create subgroups for each theme
        X_parts_list = []
        bckg_parts = []
        for i in range(len(self.themes)):
            if len(X_sum_list[i]) >= self.n_topics:
                X_sum_list[i].sort(reverse=True) # sort lists according to tf_scores
                X_sum_list[i] = np.array(X_sum_list[i])
                if len(X_sum_list[i]) >= 2*self.n_topics:
                    X_sum_list[i] = np.array(X_sum_list[i][:len(X_sum_list[i])//2]) # take half of the most dense documents from each theme
                X_parts_list.append(partition_list(X_sum_list[i], self.n_topics)) # partition each list into n_topics 
                bckg_parts.extend(X_sum_list[i][:len(X_sum_list[i])//4, 1]) # Take most dense quarter of the documents from each theme
            else:
                # partition each list into n_topics so that each partition can have p documents
                X_parts_list.append([np.array(random.choices(X_sum, k=len(self.train_data)//len(self.themes)//2//3+1)) for j in range(self.n_topics)])
                
        bckg_inital_matrix = np.average(X[bckg_parts], axis=0) # Background part has the same vector for each theme
        
        # Assign each subgroups average for each theme to the corresponding theme-related row of H
        if self.separate_models == True:
            H_list = []
            
            for i in range(len(self.themes)):
                H = np.zeros((self.n_topics+1, X.shape[1]))
                for j in range(self.n_topics):
                    H[j] = np.average(X[X_parts_list[i][j%len(X_parts_list[i])][:, 1]], axis=0)
                H[self.n_topics] = bckg_inital_matrix.copy()
                H_list.append(H)
                
            return H_list
        else:
            H = np.zeros((len(self.themes)*self.n_topics+1, X.shape[1]))
            for i in range(len(self.themes)):
                for j in range(self.n_topics):
                    H[i*self.n_topics + j] = np.average(X[X_parts_list[i][j%len(X_parts_list[i])][:, 1]], axis=0)
                H[len(self.themes)*self.n_topics] = bckg_inital_matrix.copy()
            return H
        
        
    def fit(self):
        '''
        Main function that fits the given data to TSNMF model.
        
        Returns a dictionary
          * 'tsnmf' / 'tsnmf_list': sklearn NMF object/list.
          * 'W' / 'W_list': Document-Theme matrix/matrices list.
          * 'tf': term frequency matrix returned from sklearn.
          * 'tf_vectorizer': term frequency vectorizer created using sklearn
        '''
        t00 = time()
        
        corpus = list(self.train_data.text)
        n_docs = len(corpus)
        # n_themes = len(self.themes)
        
        t0 = time()
        tf, tf_vectorizer, n_features = self.term_vectorization(corpus)
        #print("done in %0.2fs." % (time() - t0))
        #print()
        
        tsnmf_context = dict()
                       
        t0 = time()
        if self.separate_models == True:
            #print("Generating W Matrices.. " , end=' ')
            W_list = self.generate_W()
            #print("done in %0.2fs." % (time() - t0))
            #print()            
            
            if self.bCool_init == True:
                #print("Generating H Matrices with bCool.. " , end=' ')
                H_list = self.bCool_initialization_H(tf)
                #print("done in %0.2fs." % (time() - t0))
                #print()
            
            tsnmf_list = []
            for i, W in enumerate(W_list):
                #print("Fitting TSNMF for " + str(self.themes[i]) , end=' ')
                t0 = time()
                
                if self.bCool_init == True:
                    H = H_list[i]
                else:
                    # scaling with 2*sqrt(X.mean() / n_components) like in random init of scikit nmf
                    H = np.random.rand(self.n_topics+1, n_features)*2*np.sqrt(tf.mean()/(self.n_topics+1))

                tsnmf = NMF(n_components= self.n_topics+1, solver='mu', beta_loss=self.beta_loss, alpha=.1, l1_ratio=.5, init = 'custom')

                W = tsnmf.fit_transform(X=tf,W=W,H=H)
                #print("done in %0.2fs." % (time() - t0))

                tsnmf_list.append(tsnmf)
                
            tsnmf_context['tsnmf_list'] = tsnmf_list
            tsnmf_context['W_list'] = W_list
            tsnmf_context['tf'] = tf
            tsnmf_context['tf_vectorizer'] = tf_vectorizer
                            
        else:
            #print("Generating W Matrix.. " , end=' ')
            W = self.generate_W()
            #print("done in %0.2fs." % (time() - t0))
            #print()
            
            #print("Fitting TSNMF", end=' ')
            
            t0 = time()
            
            if self.bCool_init == True:
                #print("Generating H Matrix with bCool.. " , end=' ')
                H = self.bCool_initialization_H(tf)
                #print("done in %0.2fs." % (time() - t0))
                #print()
            else:
                # len(self.themes)*self.n_topics+1 -> W.shape[1]
                # scaling with 2*sqrt(X.mean() / n_components) like in random init of scikit nmf
                H = np.random.rand(W.shape[1], n_features)*2*np.sqrt(tf.mean()/W.shape[1])

            tsnmf = NMF(n_components= W.shape[1], solver='mu', beta_loss=self.beta_loss, alpha=.1, l1_ratio=.5, init = 'custom')
            
            W = tsnmf.fit_transform(X=tf,W=W,H=H)
            #print("done in %0.2fs." % (time() - t0))
            
            tsnmf_context['tsnmf'] = tsnmf
            tsnmf_context['W'] = W
            tsnmf_context['tf'] = tf
            tsnmf_context['tf_vectorizer'] = tf_vectorizer
        
        #print("ALL DONE in %0.2fs." % (time() - t00))
        return tsnmf_context 
    
    
    def transform_test_corpus(self, tsnmf, tf_vectorizer, word_replacements=[]):
        '''
        Transforms test corpus to Document-Theme Matrix using trained TSNMF models.
        
        Parameters
        ----------
        tsnmf_context: dictioanry returned from self.fit()
        word_replacements: Synonym apllication to be able too cover as much word from training data as possible in the test data. 
            Replaces test words synonyms if possible where the synonyms are taken from training dictionary. 
        '''
        if word_replacements == []:
            tf_test = tf_vectorizer.transform(list(self.test_data['text']))
        else:
            docs = [syn.replace_synoyms(list(self.test_data['text'])[idx], word_replacements[idx][i]) for idx in range(len(list(self.test_data['text'])))]
            tf_test = tf_vectorizer.transform(docs)

        W_test = tsnmf.transform(tf_test)
        return W_test, tf_test
    
    
    def evaluate_test_corpus(self, tsnmf_context, word_replacements=[]):
        '''
        Transforms test corpus to Document-Theme Matrix using trained TSNMF models and evaluates the theme scoring matrices of the test data.
        
        Parameters
        ----------
        tsnmf_context: dictioanry returned from self.fit()
        word_replacements: Synonym apllication to be able too cover as much word from training data as possible in the test data. 
            Replaces test words synonyms if possible where the synonyms are taken from training dictionary. 
        
        Returns modified tsnmf_context with additional fields:
        ----------
        W_test_high: Maximum theme score of each test documents.
        W_test_norms: All theme scores of each test document.
        W_test/list (partial): Document-Theme matrix of each test document / for each theme (partial).
        tf_test: Term frequency matrix returned from sklearn.
        '''
        
        t0 = time()
        
        if self.supervision == 'supervised':
            #print("Transforming Test Corpus.. ")
            if self.separate_models == True:
                W_test_list = []
                for i, tsnmf in enumerate(tsnmf_context['tsnmf_list']):
                    #print("Transforming for " + str(self.themes[i]))
                    W_test, tf_test = self.transform_test_corpus(tsnmf, tsnmf_context['tf_vectorizer'], word_replacements=[])
                    W_test_list.append(W_test)
            else:
                W_test, tf_test = self.transform_test_corpus(tsnmf_context['tsnmf'], tsnmf_context['tf_vectorizer'], word_replacements=[])


        if self.separate_models == True:
            if self.supervision == 'semi_supervised':
                W_test_list = []
                for W_test in tsnmf_context['W_list']:
                    W_test_list.append(W_test[list(self.test_data.index)])
                tf_test = tsnmf_context['tf']
                
            # Supervised Partial
            W_test_norms = []
            for W_test in W_test_list:
                temp_docs = []
                for dd in W_test:
                    temp = []
                    for w in dd[:-1]:
                        if self.background_scoring:
                            temp.append(100*w/(w+dd[-1]))
                        else:
                            temp.append(w)
                    temp_docs.append(temp)
                W_test_norms.append(temp_docs)

            W_test_norms = np.asarray(W_test_norms)
            W_test_norms = np.nan_to_num(W_test_norms)

            W_test_high = W_test_norms.max(axis=2).T
            
            tsnmf_context['W_test_high'] = W_test_high
            tsnmf_context['W_test_norms'] = W_test_norms
            tsnmf_context['W_test_list'] = np.asarray(W_test_list)
            tsnmf_context['tf_test'] = tf_test
                    
            #return W_test_high, W_test_norms, np.asarray(W_test_list), tf_test
        
        else:
            if self.supervision == 'semi_supervised':
                W_test = tsnmf_context['W'][list(self.test_data.index)]
                tf_test = tsnmf_context['tf']
                
            W_test_norms = []        
            for t in range(len(self.themes)):
                temp_docs = []
                for dd in W_test:
                    temp = []
                    for w in dd[self.n_topics*t : self.n_topics*(t+1)]:
                        if self.background_scoring:
                            temp.append(100*w/(w+dd[-1]))
                        else:
                            temp.append(w)
                    temp_docs.append(temp)
                W_test_norms.append(temp_docs)

            W_test_norms = np.asarray(W_test_norms)
            W_test_norms = np.nan_to_num(W_test_norms)

            W_test_high = W_test_norms.max(axis=2).T
            
            tsnmf_context['W_test_high'] = W_test_high
            tsnmf_context['W_test_norms'] = W_test_norms
            tsnmf_context['W_test'] = np.asarray(W_test)
            tsnmf_context['tf_test'] = tf_test

            #return W_test_high, W_test_norms, np.asarray(W_test), tf_test
            
        #print("ALL DONE in %0.2fs." % (time() - t0))
        
        return tsnmf_context
    
    
    
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


# Shows the distribution of labels across real data, train and test
# def calculate_theme_dist(document_labels_list):
#     '''
#     document_labels_list: list of lists that includes labels of each document 
#     '''

#     counter_dict = {i:0 for i in themes_ids}
#     total_theme_count = 0
#     for doc_themes in document_labels_list:
#         for theme in doc_themes:
#             counter_dict[theme] += 1
#             total_theme_count += 1
#     for k, v in counter_dict.items():
#         counter_dict[k] /= total_theme_count
    
#     return counter_dict
    
    
# counter_dict_data = calculate_theme_dist(doc_theme_ids)
# counter_dict_train = calculate_theme_dist(stratified_data[0])
# counter_dict_test = calculate_theme_dist(stratified_data[1])


# lists_data = sorted(counter_dict_data.items()) # sorted by key, return a list of tuples
# x_data, y_data = zip(*lists_data) # unpack a list of pairs into two tuples

# lists_train = sorted(counter_dict_train.items()) # sorted by key, return a list of tuples
# x_train, y_train = zip(*lists_train) # unpack a list of pairs into two tuples

# lists_test = sorted(counter_dict_test.items()) # sorted by key, return a list of tuples
# x_test, y_test = zip(*lists_test) # unpack a list of pairs into two tuples

# plt.figure()
# plt.plot(x_data, y_data, '--')
# plt.plot(x_train, y_train, '-+')
# plt.plot(x_test, y_test, '-*')
# plt.show()


def partition_list(a, k):
    '''
    Partition list a into k partitions
    https://stackoverflow.com/a/35518205
    Slightly modified to cope with oru method. Input a is a list of tuples
    where the first index is the value and the second index is the id.
    Modifications are commented as Modified 
    
    This approach defines partition boundaries that divide the array in roughly equal numbers of elements,
    and then repeatedly searches for better partitionings until it can't find any more.
    It differs from most of the other posted solutions in that it looks to find an optimal solution
    by trying multiple different partitionings. The other solutions attempt to create a good partition
    in a single pass through the array, but I can't think of a single pass algorithm that's guaranteed optimal.
    '''
    
    #check degenerate conditions
    # Modified: if there is not enough item in list a
    if len(a) == 0: return np.array([])
    if k >= len(a): return [np.array([random.choice(a)]) for i in range(k)]
    #if k >= len(a): return np.array([[x] for x in a])
    if k <= 1: return np.array([a])
    
    
    #create a list of indexes to partition between, using the index on the
    #left of the partition to indicate where to partition
    #to start, roughly partition the array into equal groups of len(a)/k (note
    #that the last group may be a different size) 
    partition_between = []
    for i in range(k-1):
        partition_between.append((i+1)*len(a)//k)
    #the ideal size for all partitions is the total height of the list divided
    #by the number of paritions
    average_height = float(sum(a[:,0]))/k # Modification a -> a[:,0]
    best_score = None
    best_partitions = None
    count = 0
    no_improvements_count = 0
    #loop over possible partitionings
    while True:
        #partition the list
        partitions = []
        index = 0
        for div in partition_between:
            #create partitions based on partition_between
            partitions.append(a[index:div])
            index = div
        #append the last partition, which runs from the last partition divider
        #to the end of the list
        partitions.append(a[index:])
        #evaluate the partitioning
        worst_height_diff = 0
        worst_partition_index = -1
        for p_ind, p in enumerate(partitions): # Modification enumerat
            #compare the partition height to the ideal partition height
            height_diff = average_height - sum(p[:,0]) # Modification p -> p[:,0]
            #if it's the worst partition we've seen, update the variables that
            #track that
            if abs(height_diff) > abs(worst_height_diff):
                worst_height_diff = height_diff
                worst_partition_index = p_ind # Modification partitions.index(p) -> p_ind
        #if the worst partition from this run is still better than anything
        #we saw in previous iterations, update our best-ever variables
        if best_score is None or abs(worst_height_diff) < best_score:
            best_score = abs(worst_height_diff)
            best_partitions = partitions
            no_improvements_count = 0
        else:
            no_improvements_count += 1
        #decide if we're done: if all our partition heights are ideal, or if
        #we haven't seen improvement in >5 iterations, or we've tried 100
        #different partitionings
        #the criteria to exit are important for getting a good result with
        #complex data, and changing them is a good way to experiment with getting
        #improved results
        if worst_height_diff == 0 or no_improvements_count > 5 or count > 100:
            return best_partitions
        count += 1
        #adjust the partitioning of the worst partition to move it closer to the
        #ideal size. the overall goal is to take the worst partition and adjust
        #its size to try and make its height closer to the ideal. generally, if
        #the worst partition is too big, we want to shrink the worst partition
        #by moving one of its ends into the smaller of the two neighboring
        #partitions. if the worst partition is too small, we want to grow the
        #partition by expanding the partition towards the larger of the two
        #neighboring partitions
        if worst_partition_index == 0:   #the worst partition is the first one
            if worst_height_diff < 0: partition_between[0] -= 1   #partition too big, so make it smaller
            else: partition_between[0] += 1   #partition too small, so make it bigger
        elif worst_partition_index == len(partitions)-1: #the worst partition is the last one
            if worst_height_diff < 0: partition_between[-1] += 1   #partition too small, so make it bigger
            else: partition_between[-1] -= 1   #partition too big, so make it smaller
        else:   #the worst partition is in the middle somewhere
            left_bound = worst_partition_index - 1   #the divider before the partition
            right_bound = worst_partition_index   #the divider after the partition
            if worst_height_diff < 0:   #partition too big, so make it smaller
                # Modifications partitions[...][:,0]
                if sum(partitions[worst_partition_index-1][:,0]) > sum(partitions[worst_partition_index+1][:,0]):   #the partition on the left is bigger than the one on the right, so make the one on the right bigger
                    partition_between[right_bound] -= 1
                else:   #the partition on the left is smaller than the one on the right, so make the one on the left bigger
                    partition_between[left_bound] += 1
            else:   #partition too small, make it bigger
                if sum(partitions[worst_partition_index-1][:,0]) > sum(partitions[worst_partition_index+1][:,0]): #the partition on the left is bigger than the one on the right, so make the one on the left smaller
                    partition_between[left_bound] -= 1
                else:   #the partition on the left is smaller than the one on the right, so make the one on the right smaller
                    partition_between[right_bound] += 1

                    
# def print_best_partition(a, k):
#     print('Partitioning {0} into {1} partitions'.format(a, k))
#     p = partition_list(a, k)
#     print('The best partitioning is {0}\n    With heights {1}\n'.format(p, list(map(sum, p))))