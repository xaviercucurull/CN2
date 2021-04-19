"""
CN2 Induction Algorithm
Author: Xavier Cucurull Salamero <xavier.cucurull@estudiantat.upc.edu>

"""

import pandas as pd
from scipy.stats import entropy
import numpy as np


class CN2():
    """ Implementation of the CN2 (Clark and Niblett 1989) Induction Algorithm.
    
    Obtain a set of ordered prediction rules from a set of examples.
    Rules are of the form:  IF <complex> THEN predict Class

    Definitions:
        - Selector: a basic test on an attribute
        - Complex: conjuction of selectors
        - Star: group of complexes
    
    Example:
        >>> cn2 = CN2()
        >>> cn2.fit(x_train, y_train)
        >>> y_pred = cn2.predict(x_test)
    """
    def __init__(self, max_star_size=5, min_significance=0.7, verbose=0):
        self.max_star_size = max_star_size
        self.min_significance = min_significance
        self.E = None
        self.bins = {}
        self.verbose = verbose
    
    def _init_selectors(self):
        """ Initialize list of selectors by getting all the possible combinations
        of feature-value pairs.
        """
        assert self.E is not None, 'E not initialized, call self.fit(x, y) first!'
        
        # treat each column (except the last one - class) as features
        for feature in self.E.columns[:-1]:
            for value in self.E[feature].unique():
                self.selectors.append({feature: value})
                
    def _find_best_complex(self):
        """ Find the best complex by iteratively specializing a star and calculating
        the quality of its complexes.
        
        At each stage in the search, CN2 retains a size-limited star of 'best complexes found so far'.

        Returns:
            list: containing best_cpx (list of dicts), best_cpx_covered_ids (list), 
                  best_cpx_most_common_class (string) and best_cpx_precision (float)
        """
        star = []
        fist_run = True
        best_cpx = None
        best_cpx_covered_ids = []
        best_cpx_most_common_class = None
        best_cpx_precision = 0
        best_entropy = 99
        best_significance = 0
        
        # while star is not empty and significance is above min
        # but don't keep iterating if a complex with significance 1 and entropy 0 is found
        while len(star) and best_significance < 1 and best_significance >= self.min_significance and best_entropy > 0 or fist_run:
            fist_run = False
            cpx_significances = []
            cpx_entropies = []
            cpx_classes = []
            
            # specialize all complexes in star
            new_star = self._specialize_star(star)
            
            # calculate entropy and significance for each complex
            for cpx in new_star:                
                # find covered examples
                covered_ids = self._find_covered_examples(cpx, self.E)
                
                # calculate class probability distrubitions only if the complex covers examples
                if len(covered_ids):
                    covered_prob_dist = self.E['class'].loc[covered_ids].value_counts(sort=False, normalize=True)
                    covered_classes = covered_prob_dist.keys()
                    try:
                        most_common_class = covered_prob_dist.sort_values(ascending=False).keys()[0]
                    except:
                        most_common_class = 'ERROR'
                        print('ERROR - covered ex: {}'.format(covered_ids))
                    cpx_classes.append(most_common_class)
                    class_prob_dist = np.array(self.E['class'].loc[self.E['class'].isin(covered_classes)].value_counts(sort=False, normalize=True))     # global prob dist of covered classes

                    # calculate complex entropy
                    covered_prob_dist = np.array(covered_prob_dist)
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
                        best_cpx_precision = np.sort(covered_prob_dist)[::-1][0]
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

            # print information about each complex of the new_star
            if self.verbose:
                print('Len star:{}'.format(len(new_star)))
                for i in range(len(new_star_df)):
                    print('{} -> {} (sig: {:.3f} - ent: {:.3f})'.format(new_star_df.iloc[i, 0], new_star_df.iloc[i, 3], new_star_df.iloc[i, 2], new_star_df.iloc[i, 1]))
                print('---------------------------------------')
            
        return best_cpx, best_cpx_covered_ids, best_cpx_most_common_class, best_cpx_precision
    
    def _specialize_star(self, star):
        """ Specialize a star by adding new conjuctive terms to its complexes.

        Args:
            star (list): list of complexes

        Returns:
            list: new specialized star
        """
        # specialize the star (subset of best complexes found so far) by adding new conjuctive terms
        new_star = []
        if len(star):
            # specialize star by adding new conjuctive terms to each complex
            for cpx in star:
                for selector in self.selectors:
                    sel_attribtute = list(selector)[0]
                    if sel_attribtute not in cpx.keys():  # if attribute not in complex (avoid null)
                        new_cpx = cpx.copy()
                        new_cpx[sel_attribtute] = selector[sel_attribtute]  # add new selector to complex
                        if new_cpx not in new_star:     # avoid repeating a complex (different order, but same meaning)
                            new_star.append(new_cpx)  
        else:
            # initialize star with all selectors as complexes
            new_star = [selector for selector in self.selectors]
        
        return new_star

    def _find_covered_examples(self, cpx, df, hasclass=True):
        """ Find all examples covered by a complex and returned
        their corresponding indices.

        Args:
            cpx (dictionary): complex used as filter of the form {'attribute': value}
            df (DataFrame): set of examples
            hasclass (boolean): specifies if the last column of the given df is the class 

        Returns:
            list: list of indices of the examples covered
        """
        # https://stackoverflow.com/a/34162576
        # find indices where dataframe matches filter dict
        if hasclass:       
            covered_ids = df.loc[(df.iloc[:, :-1][list(cpx)] == pd.Series(cpx)).all(axis=1)]
        else:
            covered_ids = df.loc[(df[list(cpx)] == pd.Series(cpx)).all(axis=1)]
        
        return covered_ids.index
    
    def fit(self, X, y, n_bins=4, fixed_bin_size=False):
        """ Fit training data and compute CN2 induction rules.
            
        Args:
            x (DataFrame): training data features
            y (array-like): training data classification
            n_bins (int, optional): number of bins used for discretization of continuous attributes. Defaults to 4.
            fixed_bin_size (boolean, optional): use a fixed size bin when discretizing. 
                                                True uses pandas.cut, False uses pandas.qcut Defaults to False.
        """
        self.E = X.copy()
        
        # Discretize continuous attributes
        for c in self.E.columns:
            if len(self.E[c].value_counts()) > n_bins:
                if 'int' in str(self.E[c].dtype):
                    precision = 0
                    if fixed_bin_size:
                        self.E[c], self.bins[c] = pd.cut(self.E[c], n_bins, precision=precision, retbins=True, duplicates='drop')
                    else:
                        self.E[c], self.bins[c] = pd.qcut(self.E[c], n_bins, precision=precision, retbins=True, duplicates='drop')

                elif 'float' in str(self.E[c].dtype):
                    precision = 2
                    if fixed_bin_size:
                        self.E[c], self.bins[c] = pd.cut(self.E[c], n_bins, precision=precision, retbins=True, duplicates='drop')
                    else:
                        self.E[c], self.bins[c] = pd.qcut(self.E[c], n_bins, precision=precision, retbins=True, duplicates='drop')
        self.E['class'] = y     # add class columns to examples DataFrame

        self.selectors = []     # list of all possible selectors
        self._init_selectors()
        
        # some class statistics, used later to calculate rule coverage
        total_class_counts = pd.Series(y).value_counts()
        
        # get global most common class, used by default rule
        default_class = total_class_counts.keys()[0]
        
        self.rules_list = []
            
        # loop until all examples are covered
        while len(self.E):
            # print examples not covered yet
            if self.verbose:
                print('Examples to cover:\n {}\n\nCandidate complexes:\n'.format(self.E))

            # find best complex
            best_cpx, best_cpx_covered_ids, best_cpx_most_common_class, best_cpx_precision = self._find_best_complex()
            
            # if best_complex not null
            if best_cpx is not None:
                # calculate rule coverage
                best_cpx_coverage = self.E['class'].loc[best_cpx_covered_ids].value_counts().iloc[0] / total_class_counts[best_cpx_most_common_class]
                                
                # remove best_cpx_covered_ids from self.E
                self.E.drop(best_cpx_covered_ids, inplace=True)

                # add (complex, class, coverage, precision) to rules list
                self.rules_list.append((best_cpx, best_cpx_most_common_class, best_cpx_coverage, best_cpx_precision))
                
                # print obtained rule
                if self.verbose:
                    print('Chosen rule:\nIF {} THEN {}  [{:.2f} {:.2f}]\n'.format(best_cpx, best_cpx_most_common_class, 
                                                                                  best_cpx_coverage, best_cpx_precision))
        
        # add default rule
        self.rules_list.append((None, default_class, 0, 0))
        
    def predict(self, X):
        """ Use the obtained induction rules to make a prediciton on the given data.

        Args:
            X (DataFrame): test data features with the same header used during training (fit)
            
        Returns:
            array-like: list of predictions
        """
        x = X.copy()
        y = pd.Series([None]*len(x))
        
        # discretize attributes using saved bins
        for c in x.columns:
            if c in self.bins.keys():
                x[c] = pd.cut(x[c], self.bins[c])

        assert len(self.rules_list), 'CN2 rules not induced, call self.fit(x, y) first!'
        
        # apply all rules in order
        for i in range(len(self.rules_list) - 1):
            cpx = self.rules_list[i][0]
            prediction = self.rules_list[i][1]
            
            # use complex from rule and use as filter
            covered_ids = self._find_covered_examples(cpx, x, hasclass=False)
            
            # update prediction Series with prediction of current rule
            y.loc[covered_ids] = prediction            
            
            # remove covered examples from x DataFrame
            x.drop(covered_ids, inplace=True)

        # apply default rule to examples not classified
        y.loc[x.index] = self.rules_list[-1][1]
        
        return np.array(y)
    
    def _generate_interpretable_rules(self):
        """ Generate a list of rules in an interpretable way.
        Rules are generated in the form of: IF <complex> THEN CLASS IS <class>
        
        This method is used by print_rules and generate_rules_table.
        
        Returns:
            list: a list of strings containing the interpretable rules
        """
        interpretable_rules = []

        for i in range(len(self.rules_list) - 1):
            cpx = self.rules_list[i][0]
            prediction = self.rules_list[i][1]
            if len(cpx) == 1:
                rule_str = 'IF {} IS {} THEN CLASS IS {}'.format(list(cpx.keys())[0], list(cpx.values())[0], prediction)
            else:
                conditions = ''
                for f, v in cpx.items():
                    conditions += ' {} IS {} AND'.format(f, v)
                conditions = conditions[:-4]    # remove final AND
                
                rule_str = 'IF{} THEN CLASS IS {}'.format(conditions, prediction)
                
            # add interpretable rule to list
            interpretable_rules.append(rule_str)

        # add default rule
        interpretable_rules.append('DEFAULT CLASS IS {}'.format(self.rules_list[-1][1]))
        
        return interpretable_rules
    
    def print_rules(self):
        """ Print all rules in an interpretable way.
        Rules are printed in the form of: IF <complex> THEN CLASS IS <class>
        """
        interpretable_rules = self._generate_interpretable_rules()
        for r in interpretable_rules:
            print(r)

    def save_rules(self, filename):
        """ Save the generated rules into a text file in an interpretable way

        Args:
            filename (str): full filename of the file where to write the rules
        """
        with open(filename, 'w') as f:
            interpretable_rules = self._generate_interpretable_rules()
            f.write('\n'.join(interpretable_rules) + '\n')

    def generate_rules_table(self):
        """ Generate a table containing the interpretable rules and their corresponding coverage and precision.
        
        The generated pandas DataFrame can then be saved to LaTeX.

        Returns:
            DataFrame: table containing rules, coverage and precision
        """
        interpretable_rules = self._generate_interpretable_rules()
        rules_coverage = ['{:.2f}'.format(i[2]*100) for i in self.rules_list]
        rules_precision = ['{:.2f}'.format(i[3]*100) for i in self.rules_list]
        
        rules_table = pd.DataFrame({'Rules': interpretable_rules, 'Coverage': rules_coverage, 
                                    'Precision': rules_precision})

        return rules_table
        