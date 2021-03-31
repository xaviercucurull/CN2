from datasets import datasets

#x, y = datasets.load_tennis()

x, y = datasets.load_lenses()


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
    def __init__(self, max_star_size=3, min_significance=0.5):
        self.max_star_size = max_star_size
        self.min_significance = min_significance
        self.x = None
        self.y = None
    
    def _init_selectors(self):
        """ Initialize list of selectors by getting all the possible combinations
        of feature-value pairs.
        """
        assert self.x is not None, 'X not initialized, call self.fit(x, y) first!'
        
        for feature in x.columns:
            for value in x[feature].unique():
                self.selectors.append({feature: value})
                
    def _find_best_complex(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        best_complex = []
        star = []
        fist_run = True
        best_significance = 1
        
        # while star is not empty and significance is above min
        while len(star) and best_significance > self.min_significance or fist_run:
            fist_run = False
            # specialize all complexes in star
            new_star = self._specialize_star(star)
            # for every complex c in new_star
                # if c is better (entropy+significance) than best_complex tested on E
                # replace best_complex by c
            # remove worst complex of new_star to get max_star_size
            star = new_star
            
            #########################################################
            # TEST
            print('Len star:{}'.format(len(star)))
            for c in star:    
                print(c)
            print('---------------------------------------')
            #########################################################
            
        return best_complex
    
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

        
    def _entropy(self, cpx):
        """ Evaluate complex entropy

        Args:
            cpx ([type]): [description]

        Returns:
            [type]: [description]
        """
        return entropy
        
    def _significance(self, cpx):
        """ Evaluate complex significance

        Args:
            cpx ([type]): [description]

        Returns:
            [type]: [description]
        """
        return significance
    
    def fit(self, X, y):
        """ Fit training data and compute CN2 induction rules

        Args:
            x (DataFrame): training data features
            y (DataFrame): training data predictions
        """
        self.x = X
        self.y = y
        
        self.E = [i for i in range(len(x))]     # use index reference
        self.E = self.x.copy()                  # list of classified examples (ALL instances)
        self.E['class'] = self.y                # add class column
        
        self.selectors = []             # list of all possible selectors
        self._init_selectors()
        
        self.rules_list = []
        
        # loop until all examples are covered
        while len(self.E):
            best_cpx = self._find_best_complex()
            # if best_complex not nil (?)
            # find examples covered by best_complex
            E_prima = 0     # filter E? use ids?
            self.E = []
        
    def predict(self, X):
        y = None
        return y
    
    
        
cn2 = CN2()

cn2.fit(x, y)
