import pandas as pd
from scipy.stats import entropy
import numpy as np


class CN2():
    """[summary]
    - Selector: a basic test on an attribute
    - Complex: conjuction of selectors
    
    if <complex> then predict C
    
    At each stage in the search, CN2 retains a size-limited set or star S of 'best complexes found so far'
    A complex is specialized by adding a conjuctive term or removing a disjunctive element in one of its selectors
    The star is then trimmed by removing lowest ranking elements
    Specialization: intersect set of possible selectors with current star -> eliminate null and unchanged elements
    """
    def __init__(self, max_star_size=5, min_significance=0.7, verbose=0):      # TODO: how does star size affect? - check various values also for min_significance
        self.max_star_size = max_star_size
        self.min_significance = min_significance
        self.E = None
        self.verbose = verbose
    
    def _init_selectors(self):
        """ Initialize list of selectors by getting all the possible combinations
        of feature-value pairs.
        """
        assert self.E is not None, 'E not initialized, call self.fit(x, y) first!'
        
        # treat each column (except the last one - class) as features
        for feature in self.E.columns[:-1]:
            for value in x[feature].unique():
                self.selectors.append({feature: value})
                
    def _find_best_complex(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        star = []
        fist_run = True
        best_cpx = None
        best_entropy = 99
        best_significance = 0
        
        # while star is not empty and significance is above min
        # but don't keep iterating if a complex with significance 1 and entropy 0 is found
        while len(star) and best_significance < 1 and best_significance > self.min_significance and best_entropy > 0 or fist_run:
            fist_run = False
            cpx_significances = []
            cpx_entropies = []
            cpx_classes = []
            
            # specialize all complexes in star
            new_star = self._specialize_star(star)
            
            # calculate entropy and significance for each complex
            for cpx in new_star:                
                # find covered examples
                covered_ids = self._find_covered_examples(cpx)
                
                # calculate class probability distrubitions only if the complex covers examples
                if True in covered_ids:
                    # print('covered_ids {}'.format(covered_ids))   # TODO REMOVE
                    # TODO: where to calculate rule coverage?
                    covered_prob_dist = self.E['class'].loc[covered_ids].value_counts(sort=False, normalize=True)
                    covered_classes = covered_prob_dist.keys()
                    try:
                        most_common_class = covered_prob_dist.sort_values(ascending=False).keys()[0]
                    except:
                        most_common_class = 'ERROR'
                        print('ERROR - covered ex: {}'.format(covered_ids))
                    cpx_classes.append(most_common_class)
                    covered_prob_dist = np.array(covered_prob_dist)
                    class_prob_dist = np.array(self.E['class'].loc[self.E['class'].isin(covered_classes)].value_counts(sort=False, normalize=True))     # global prob dist of covered classes

                    # calculate complex entropy
                    cpx_ent = entropy(covered_prob_dist)
                        
                    # calculate complex significance as (1 - likelihood ratio statistic)
                    cpx_sig = 1 - np.multiply(covered_prob_dist, np.log(np.divide(covered_prob_dist, class_prob_dist))).sum() * 2
                        
                    # add metrics to list
                    cpx_entropies.append(cpx_ent)
                    cpx_significances.append(cpx_sig)
                    
                    # check if cpx is better than best_cpx
                    if cpx_sig >= best_significance and cpx_ent < best_entropy:
                        best_significance = cpx_sig
                        best_entropy = cpx_ent
                        best_cpx = cpx
                        best_cpx_covered_ids = covered_ids
                        best_cpx_most_common_class = most_common_class
                        best_cpx_class_perc = covered_prob_dist
                else:
                    cpx_entropies.append(99)
                    cpx_significances.append(0)
                    cpx_classes.append(None)
                    
            # create dataframe to easily sort
            new_star_df = pd.DataFrame({'complex': new_star, 'entropy': cpx_entropies, 'significance': cpx_significances, 'class': cpx_classes})
            # sort by complex quality (higher significance and lower entropy)
            # and remove worst complexes from new_star to keep max_star_size
            new_star_df = new_star_df.sort_values(by=['significance', 'entropy'], 
                                                  ascending=[False, True]).iloc[:self.max_star_size]
            
            star = new_star_df.complex.to_list()

            # TODO remove - just test / verbose
            if self.verbose:
                print('Len star:{}'.format(len(new_star)))
                for i in range(len(new_star_df)):
                    print('{} -> {} (sig: {:.3f} - ent: {:.3f})'.format(new_star_df.iloc[i, 0], new_star_df.iloc[i, 3], new_star_df.iloc[i, 2], new_star_df.iloc[i, 1]))
                print('---------------------------------------')
        
        print(best_cpx_class_perc)
            
        return best_cpx, best_cpx_covered_ids, best_cpx_most_common_class
    
    def _specialize_star(self, star):
        # specialize the star (subset of best complexes found so far) by adding new conjuctive terms
        new_star = []
        if len(star):
            # specialize star by adding new conjuctive terms to each complex
            for cpx in star:
                for selector in self.selectors:
                    sel_attribtute = list(selector)[0]
                    if sel_attribtute not in cpx.keys():  # if attribute not in complex (avoid null)
                        new_cpx = cpx.copy()
                        new_cpx[sel_attribtute] = selector[sel_attribtute]
                        if new_cpx not in new_star:     # avoid repeating a complex (different order, but same meaning)
                            new_star.append(new_cpx)  
            
            #new_star = []   # REMOVE, JUST TO TEST

        else:
            # initialize star with all selectors as complexes
            new_star = [selector for selector in self.selectors]
        
        return new_star

    def _find_covered_examples(self, cpx):
        """ Find all examples covered by a complex and returned
        their corresponding indices.

        Args:
            cpx (dictionary): complex used as filter of the form {'attribute': value}

        Returns:
            list: list of indices of the examples covered
        """
        # https://stackoverflow.com/a/34162576
        # find indices where dataframe matches filter dict       
        covered_ids = np.array((self.E.iloc[:, :-1][list(cpx)] == pd.Series(cpx)).all(axis=1))
        
        return np.array(covered_ids)
    
    def _remove_covered_examples(self, covered_examples_ids):
        """[summary]

        Args:
            covered_examples_ids ([type]): [description]
        """
        self.E.drop(np.where(covered_examples_ids)[0], inplace=True)
        self.E.reset_index(drop=True, inplace=True)
        
    def fit(self, X, y):
        """ Fit training data and compute CN2 induction rules

        Args:
            x (DataFrame): training data features
            y (DataFrame): training data predictions
        """
        self.E = X
        self.E['class'] = y
        self.E.reset_index(drop=True, inplace=True)     # start index from 0
        
        self.selectors = []             # list of all possible selectors
        self._init_selectors()
        
        # get global most common class, used by default rule
        # TODO
        
        # some class statistics, used later to calculate rule coverage
        # TODO
        
        self.rules_list = []
            
        # loop until all examples are covered
        while len(self.E):
            if self.verbose:
                print('Examples to cover:\n {}\n\nCandidate complexes:\n'.format(self.E))
                         
            best_cpx, best_cpx_covered_ids, best_cpx_most_common_class = self._find_best_complex()
            
            # if best_complex not null
            if best_cpx is not None:
                # remove best_cpx_covered_ids from self.E
                self._remove_covered_examples(best_cpx_covered_ids)
                # add complex, class to rules list
                # TODO (add rule in the form of 'IF best_cpx THEN THE CLASS IS best_cpx_common_class') -> this cam be done in a print rules function
                self.rules_list.append((best_cpx, best_cpx_most_common_class))
                
                if self.verbose:
                    print('Chosen rule:\nIF {} THEN {}\n'.format(best_cpx, best_cpx_most_common_class))
        
    def predict(self, X):
        y = None
        # TODO: use complex from rule and use as filter
        return y


# TODO: move to another file. Example/test/whatever
from datasets import datasets

#x, y = datasets.load_tennis()
x, y = datasets.load_lenses()

cn2 = CN2(verbose=1)

cn2.fit(x, y)
