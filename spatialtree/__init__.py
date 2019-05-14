#!/usr/bin/env python
'''
CREATED:2011-11-11 13:53:46 by Brian McFee <bmcfee@cs.ucsd.edu>

Implementation of spatial trees:
    * Max-variance KD
    * PCA tree
    * 2-means tree
    * RP tree

Also supports spill trees.

See: docs for spatialtree.spatialtree
'''

import sys
sys.setrecursionlimit(1000000000)
import numpy
import scipy.stats
import random
import heapq
from spn.algorithms.TransformStructure import Prune,Compress
from spn.algorithms.Validity import is_valid
from spn.structure.Base import Product, Sum, assign_ids, rebuild_scopes_bottom_up
from spn.structure.leaves.parametric.Parametric import create_parametric_leaf
from spn.algorithms.splitting.RDC import get_split_cols_RDC_py, get_split_cols_RDC_py
from collections import deque


class NODE_TYPE:
    SUM_NODE= 0;
    PRODUCT_NODE = 1;
    LEAF_NODE=3

class SPNRPBuilder(object):
    
    tasks =list()
    def __init__(self, data,spn_object=None,ds_context=None,leaves_size=8000,scope=None,threshold=0.4,prob=0.7,ohe=True,proportion=0.2,**kwargs):
        self.data = data
        self.spn_object = spn_object
        self.ds_context = ds_context
        self.leaves_size = leaves_size
        self.scope = scope
        self.threshold = threshold
        self.prob = prob
        self.ohe = ohe
        self.proportion = proportion

        self.root= spatialtree(data,spn_object=self.spn_object,ds_context=self.ds_context,leaves_size=self.leaves_size,scope=self.scope,threshold=self.threshold,prob=self.prob,ohe=self.ohe,proportion=self.proportion,**kwargs)
        SPNRPBuilder.tasks.append([self.root,data,kwargs])

    
    def build_spn(self):
        while SPNRPBuilder.tasks:
            sp,data,kwargs =  SPNRPBuilder.tasks.pop();
            sp.split(data,**kwargs)
            print(kwargs)


        return self.root




class spatialtree(object):

    def get_kwargs(self):
        return kwargs

    def update_ids(self):
        assign_ids(self.spn_node)
        rebuild_scopes_bottom_up(self.spn_node)
        self.spn_node = Prune(self.spn_node)

    def __init__(self, data,spn_object=None,ds_context=None,leaves_size=8000,scope=None,threshold=0.4,prob=0.7,ohe=True, **kwargs):
        self.prob = prob
        self.leaves_size = leaves_size
        self.spn_node = spn_object
        tasks = deque()
        '''
        T = spatialtree(    data, 
                            rule='kd', 
                            spill=0.25, 
                            height=H, 
                            indices=(index1, index2,...), 
                            min_items=64,
                            steps_2means=1000,
                            samples_rp=10)
                            

        Required arguments:
            data:           n-by-d data matrix (numpy.ndarray), one point per row
                            alternatively, may be a dict of vectors

        Optional arguments:
            rule:           must be one of 'kd', 'pca', '2-means', 'rp'

            spill:          what fraction of the data should propagate to both children during splits
                            must lie in range [0,1)

                            Setting spill=0 yields a partition tree

            height>0:       maximum-height to build the tree
                            default is calculated to yield leaves with ~500 items each

            indices:        list of keys/indices to store in this (sub)tree
                            default: 0:n-1, or data.keys()

            min_items:      the minimum number of items required to split a node

        Split-specific:
            steps_2means:   minimum number of steps for building the 2-means tree

            samples_rp:     number of directions to consider for each RP split
        '''

        # Default values



        if scope is None:
            scope = list(set(list(range(0,data.shape[1]))))

        self.split_cols =  get_split_cols_RDC_py(rand_gen=numpy.random.RandomState(17), ohe=ohe,threshold=threshold, n_jobs=5)
        self.spn_node = spn_object #sph_node
        self.scope = scope; #scope
        self.ds_context = ds_context

     

    def split(self, data,**kwargs):
        if 'indices' not in kwargs:
            if isinstance(data, dict):
                kwargs['indices']   = data.keys() #data 
            else:
                kwargs['indices']   = range(len(data))
            pass

        if 'proportion' in kwargs:
            self.proportion = kwargs['proportion'] #proportion
         
        n = len(kwargs['indices'])

        # Use maximum-variance kd by default
        if 'rule' not in kwargs:
            kwargs['rule'] = 'rp'
            pass

        kwargs['rule'] = kwargs['rule'].lower()



        # By default, 25% of items propagate to both subtrees
        if 'spill' not in kwargs:
            kwargs['spill']         = 0.75
            pass

        if kwargs['spill'] < 0.0 or kwargs['spill'] >= 1.0:
            raise ValueError('spill=%.2e, must lie in range [0,1)' % kwargs['spill'])

        if 'height' not in kwargs:
            print("height")
                # This calculates the height necessary to achieve leaves of roughly 500 items,
                # given the current spill threshold
            kwargs['height']    = max(0, int(numpy.ceil(numpy.log(n / self.leaves_size) / numpy.log(2.0 / (1 + kwargs['spill'])))))


        if 'min_items' not in kwargs:
            kwargs['min_items']     = 5
            pass

        if kwargs['rule'] == 'rp' and 'samples_rp' not in kwargs:
            kwargs['samples_rp']    = 10
            pass


        # All information is now contained in kwargs, we may proceed

        # Store bookkeeping information
        self.__indices      = set(kwargs['indices']) #remeber indices

        self.__splitRule    = kwargs['rule'] 
        self.__spill        = kwargs['spill']
        self.__children     = None
        self.__w            = None
        self.__thresholds   = None
        self.__keyvalue     = isinstance(data, dict) #

        # Compute the dimensionality of the data
        # This way supports opaque key-value stores as well as numpy arrays

        # Split the new node
        for x in self.__indices:
            self.__d = len(data[x])
            break
    
        '''
        Recursive algorithm to build a tree by splitting data.

        Not to be called externally.
        '''

        # First, find the split rule
    
        if kwargs['rule'] == 'pca':
            splitF  =   self.__PCA
        elif kwargs['rule'] == 'kd':
            splitF  =   self.__KD
        elif kwargs['rule'] == '2-means':
            splitF  =   self.__2means
        elif kwargs['rule'] == 'rp':
            splitF  =   self.__RP
        else:
            raise ValueError('Unsupported split rule: %s' % kwargs['rule'])

        # next stage option


        if 'NODE_TYPE'  not in kwargs:
            self.spn_node = self.produce_node(NODE_TYPE.SUM_NODE,data,self.scope)
            self.spn_node.scope.extend(self.scope)
            kwargs['NODE_TYPE'] = NODE_TYPE.SUM_NODE


        elif kwargs['NODE_TYPE'] == NODE_TYPE.SUM_NODE:

            kwargs['NODE_TYPE']= NODE_TYPE.PRODUCT_NODE #temporary

        elif kwargs['NODE_TYPE'] == NODE_TYPE.PRODUCT_NODE:
            kwargs['NODE_TYPE'] = NODE_TYPE.SUM_NODE


        if kwargs['height'] == 1:

            kwargs['NODE_TYPE'] = NODE_TYPE.LEAF_NODE

        #---Next stage option endsitems'])


        if kwargs['height'] < 0:
            raise ValueError('spatialtree.split() called with height<0')

        # If the height is 0, or the set is too small, then we don't need to split
        if kwargs['height'] == 0 or len(kwargs['indices']) < kwargs['min_items']:
            return  0
        # Compute the split direction 
        self.__w = splitF(data, **kwargs)


        # Project onto split direction
        wx = {}
        for i in self.__indices:
            wx[i] = numpy.dot(self.__w, data[i])
            pass



        # Compute the bias points
        self.__thresholds = scipy.stats.mstats.mquantiles(list(wx.values()), [self.prob - self.__spill/2, (1-self.prob) + self.__spill/2])
        # Partition the data
        left_set    = set()
        right_set   = set()

        right_data = list();
        left_data = list();



        for (i, val) in wx.items():
            if val >= self.__thresholds[0]:
                right_set.add(i)
                right_data.append(data[i])

        for (i, val) in wx.items():   
            if val < self.__thresholds[-1]:
                left_set.add(i)
                left_data.append(data[i])
            pass


        print("LEFT:"+str(len(left_set))+"Right_set:"+str(len(right_set)))

        del wx  # Don't need scores anymore

        # Construct the children
        self.__children     = list()
        total = len(left_set) + len(right_set)   
        height=kwargs['height'] 
        print(height)
        if len(self.scope) == 1:
            #self.spn_node.children.append(self.build_spn_leaf(data,scope))
            return 0;



        if kwargs['NODE_TYPE'] == NODE_TYPE.SUM_NODE or kwargs['NODE_TYPE'] == NODE_TYPE.LEAF_NODE:
            self.__children.append(self.build_node(data,left_set,len(left_set)/float(total),0,height-1,self.scope,**kwargs))
            self.__children.append(self.build_node(data,right_set,len(right_set)/float(total),1,height-1,self.scope,**kwargs))
        # Done

        elif kwargs['NODE_TYPE'] == NODE_TYPE.PRODUCT_NODE:
            print('called')
            self.__children.extend(self.build_product_node(data,left_set,len(left_set)/float(total),0,height-1,self.scope,**kwargs))
            self.__children.extend(self.build_product_node(data,right_set,len(right_set)/float(total),1,height-1,self.scope,**kwargs))

        print(self.__children)




    def produce_node(self, TYPE,data,scope):
        '''
        Update tree structure parent node type based on node type
        '''
        if  TYPE == NODE_TYPE.SUM_NODE:
            spn_node = Sum();
            spn_node.scope.extend(scope)
            return spn_node

        elif TYPE == NODE_TYPE.LEAF_NODE:
            return self.build_spn_leaf(data,scope)

    def build_node(self,data,data_set,sum_weight,spn_index,current_height,scope,**kwargs):
        kwargs['height']    =current_height
        kwargs['indices']   = data_set
        kwargs['proportion'] =sum_weight
        scope= list(set(scope))
        children = self.produce_node(kwargs['NODE_TYPE'],data,scope)

        self.spn_node.children.append(children)
        self.spn_node.weights.append(kwargs['proportion'])
        spatial_tree= spatialtree(numpy.array(data),children,ds_context = self.ds_context, **kwargs)
        print('ADDED')
        SPNRPBuilder.tasks.append([spatial_tree,data,kwargs])
        return spatial_tree


        # product node
    def build_product_node(self,data,data_set,sum_weight,children_pos,current_height,scope,**kwargs):
        scope_list = list()
        kwargs['height']    =current_height
        kwargs['indices']   = data_set
        kwargs['proportion'] =sum_weight
        node= Product();
        rptree = list()
        child_count = 0;
        for data_slice, scope_slice, _ in self.split_cols(data, self.ds_context, scope):
            if len(scope_slice) == 1 and len(data_slice) !=0:
                children = self.produce_node(NODE_TYPE.LEAF_NODE,data,scope_slice)
                node.scope.extend(scope_slice) 
                node.children.append(children)
                child_count = child_count + 1

            elif len(data_slice) !=0:
                children = self.produce_node(NODE_TYPE.SUM_NODE,data,scope_slice)
                node.scope.extend(scope_slice) 
                node.children.append(children)
                kwargs['NODE_TYPE'] = NODE_TYPE.SUM_NODE
                child_count = child_count + 1
                spatial_tree = spatialtree(numpy.array(data_slice),children,scope=scope_slice,ds_context = self.ds_context, **kwargs)
                rptree.append(spatial_tree)
                SPNRPBuilder.tasks.append([spatial_tree,data_slice,kwargs])
        if len(data_set) != 0 or child_count > 0:
            self.spn_node.children.append(node)
            self.spn_node.weights.append(sum_weight) 
        print('length')
        print(len(rptree))
        return rptree



    def build_spn_leaf(self,data,scope):
        if self.spn_node is Sum:
            return self.build_spn_leaf_product(data,scope)
        else:
            return self.build_spn_leaf_product(data,scope)

    def getIndices(self):
        return self.__indices;
    def spn_node_object(self):
        return self.spn_node;

    def build_spn_leaf_sum(self, data,scope):

        weights = 1.0/len(scope)
        spn_node = Sum()
        scope = list(set(scope))
        indices = self.getIndices()
        node_info = data[list(indices),:]
        for i in range(0,len(scope)):
            node = create_parametric_leaf(node_info[:,i].reshape(-1,1), self.ds_context, [scope[i]])
            spn_node.children.append(node)
            spn_node.scope.extend(scope)
            spn_node.weights.append(weights)

        return spn_node
                


    def build_spn_leaf_product(self, data,scope):

        spn_node = Product()
        scope = list(set(scope))
        indices = self.getIndices()
        node_info = data[list(indices),:]
        for i in range(0,len(scope)):
            node = create_parametric_leaf(node_info[:,i].reshape(-1,1), self.ds_context, [scope[i]])
            spn_node.children.append(node)

        return spn_node

                

    def update(self, D):
        '''
        T.update({new_key1: new_vector1, [new_key2: new_vector2, ...]})
        Add new data to the tree.  Note: this does not rebalance or split the tree.
        Only valid when using key-value stores.
        '''

        if not self.__keyvalue:
            raise TypeError('update method only supported when using key-value stores')

        self.__indices.update(D.keys())

        if self.isLeaf():
            return

        left_set    = {}
        right_set   = {}
        for (key, vector) in D.iteritems():
            wx = numpy.dot(self.__w, vector)

            if wx >= self.__thresholds[0]:
                right_set[key]  = vector
            if wx < self.__thresholds[-1]:
                left_set[key]   = vector
            pass

        self.__children[0].update(left_set)
        self.__children[1].update(right_set)

        pass

    # Getters and container methods
    def getHeight(self):
        '''
        Returns the height of the tree.
        A tree with no children (a leaf) has height=0.
        Otherwise, height = 1 + max(height(left child), height(right child))
        '''
        return self.__height

    def getRule(self):
        '''
        Returns the splitting rule used to generate this tree.  
        
        One of: 'kd', 'pca', '2-means', 'rp'
        '''
        return self.__splitRule

    def getSpill(self):
        '''
        Returns the spill percentage used to generate this tree.
        Floating point number in range: [0, 1)
        '''
        return self.__spill

    def getSplit(self):
        '''
        Returns the split rule for this node: a tuple (w, (lower_threshold, upper_threshold))
        where w is a vector, and the thresholds are scalars.
        For a vector x, 
            if numpy.dot(w, x) >= lower_threshold  then  x propagates to right subtree
            if numpy.dot(w, x) <  upper_threshold  then  x propagates to left subtree
        '''
        return (self.__w, self.__thresholds)

    def getDimension(self):
        '''
        Returns the dimensionality of data in this tree.
        '''
        return self.__d

    def isLeaf(self):
        '''
        Returns true if this tree is a leaf (no children)
        '''
        return self.__height == 0

    def __len__(self):
        '''
        Returns the number of data points contained in this tree.
        '''
        return len(self.__indices)

    def __contains__(self, item):
        '''
        Returns true if the given item is contained in this tree, false otherwise.
        '''
        return item in self.__indices

    def __iter__(self):
        '''
        Iterator over items contained in this tree.
        '''
        return self.__indices.__iter__()

    def traverse(self):
        '''
        Iterator over nodes in the tree.  In-order traversal.
        '''

        if self.isLeaf():
            yield self
        else:
            for t in self.__children[0].traverse():
                yield t
            yield self
            for t in self.__children[1].traverse():
                yield t
            pass

        pass

    def remove(self, x):
        '''
        Remove an item from the tree
        '''

        if x not in self.__indices:
            raise KeyError(x)

        if not self.isLeaf():
            for c in self.__children:
                if x in c:
                    c.remove(x)
            
        self.__indices.remove(x)
        pass

    # RETRIEVAL CODE

    def retrievalSet(self, **kwargs):
        '''
        S = T.retrievalSet(index=X, vector=X)
        
        Compute the retrieval set for either a given query index or vector.
        Exactly one of index or data must be supplied.
        '''

        def __retrieveIndex(idx):

            S = set()
        
            if idx in self.__indices:
                if self.isLeaf():
                    S = self.__indices.difference([idx])
                else:
                    for c in self.__children:
                        if idx in c:
                            S |= c.retrievalSet(index=idx)
                    pass
                pass

            return S

        def __retrieveVector(vec):

            S = set()

            # Did we land at a leaf?  Must be done
            if self.isLeaf():
                S = self.__indices
            else:
                Wx = numpy.dot(self.__w, vec)

                # Should we go right?
                if Wx >= self.__thresholds[0]:
                    S |= self.__children[1].retrievalSet(vector=vec)
                    pass

                # Should we go left?
                if Wx < self.__thresholds[-1]:
                    S |= self.__children[0].retrievalSet(vector=vec)
                    pass

            return S

        if 'index' in kwargs:
            if kwargs['index'] not in self:
                raise KeyError(kwargs['index'])
            return __retrieveIndex(kwargs['index'])
        elif 'vector' in kwargs:
            return __retrieveVector(kwargs['vector'])

        raise Exception('spatialtree.retrievalSet must be supplied with either an index or a data vector')
        pass



    def k_nearest(self, data, **kwargs):
        '''
        neighbors = T.k_nearest(data, k=10, index=X, vector=X)
        data:       the data matrix/dictionary
        k:          the number of (approximate) nearest neighbors to return
        index=X:    the index of the query point OR
        vector=X:   a data vector to query against
        Returns:
        A sorted list of the indices of k-nearest (approximate) neighbors of the query
        '''


        if 'k' not in kwargs:
            raise Exception('k_nearest called with no value of k')

        if not isinstance(kwargs['k'], int):
            raise TypeError('k_nearest must be called with an integer value of k')
        if kwargs['k'] < 1:
            raise ValueError('k must be a positive integer')

        # Get the retrieval set
        if 'index' in kwargs:
            x = data[kwargs['index']]
        else:
            x = kwargs['vector']
            pass

        # Now compute distance from query point to the retrieval set
        def dg(S):
            for i in S:
                yield (numpy.sum((x-data[i])**2), i)
            pass

        # Pull out indices in sorted order
        return [i for (d,i) in heapq.nsmallest(kwargs['k'], dg(self.retrievalSet(**kwargs)))]

    # PRUNING

    def prune(self, max_height):
        '''
        Prune the tree to height <= max_height.
        max_height must be a non-negative integer.
        '''
        
        if not isinstance(max_height, int):
            raise TypeError('max_height must be a non-negative integer.')
        if max_height < 0:
            raise ValueError('max_height must be a non-negative integer.')

        # If we're already a leaf, nothing to do
        if self.__height == 0:
            return

        # If max_height is 0, prune here
        if max_height == 0:
            self.__height       = 0
            self.__w            = None
            self.__children     = None
            self.__thresholds   = None
            return

        # Otherwise, recursively prune
        self.__children[0].prune(max_height - 1)
        self.__children[1].prune(max_height - 1)
        self.__height           = 1 + max(self.__children[0].getHeight(), self.__children[1].getHeight())
        pass

    # SPLITTING RULES

    def __PCA(self, data, **kwargs):
        '''
        PCA split:
        Computes a split direction by the top principal component
        (leading eigenvector of the covariance matrix) of data in 
        the current node.
        '''
        # first moment
        moment_1 = numpy.zeros(self.__d)

        # second moment
        moment_2 = numpy.zeros((self.__d, self.__d))

        # Compute covariance matrix
        for i in self.__indices:
            moment_1 += data[i]
            moment_2 += numpy.outer(data[i], data[i])
            pass

        # the mean
        moment_1    /= len(self)

        # the covariance
        sigma       = (moment_2 - (len(self) * numpy.outer(moment_1, moment_1))) / (len(self)- 1.0)

        # eigendecomposition
        (l, v)      = numpy.linalg.eigh(sigma)
        
        # top eigenvector
        w           = v[:,numpy.argmax(l)]
        return w

    def __KD(self, data, **kwargs):
        '''
        KD split:
        Finds the coordinate axis with highest variance of data
        in the current node
        '''
        moment_1 = numpy.zeros(self.__d)
        moment_2 = numpy.zeros(self.__d)

        for i in self.__indices:
            moment_1 += data[i]
            moment_2 += data[i] ** 2
            pass

        # mean
        moment_1    /= len(self)

        # variance
        sigma       = (moment_2 - (len(self) * moment_1**2)) / (len(self) - 1.0)

        # the coordinate of maximum variance
        w           = numpy.zeros(self.__d)
        w[numpy.argmax(sigma)] = 1
        return w

    def __2means(self, data, **kwargs):
        '''
        2-means split
        Computes a split direction by clustering the data in the current node
        into two, and choosing the direction spanned by the cluster centroids:
            w <- (mu_1 - mu_2)
        The cluster centroids are found by an online k-means with the Hartigan
        update.  The algorithm runs through the data in random order until
        a specified minimum number of updates have occurred (default: 1000).
        '''
        def D(u,v):
            return numpy.sum( (u-v)**2 )

        centers     = numpy.zeros( (2, self.__d) )
        counters    = [0] * 2

        index       = list(self.__indices)
        count       = 0
        num_steps   = max(len(self), kwargs['steps_2means'])

        while True:
            # Randomly permute the index
            random.shuffle(index)
            
            for i in index:
                # Find the closest centroid
                j_min = numpy.argmin([D(data[i], mu) * c / (1.0 + c) for (mu, c) in zip(centers, counters)])

                centers[j_min] = (centers[j_min] * counters[j_min] + data[i]) / (counters[j_min]+1)
                counters[j_min] += 1

                count += 1
                if count > num_steps:
                    break
                pass
            if count > num_steps:
                break

        w = centers[0] - centers[1]

        w /= numpy.sqrt(numpy.sum(w**2))
        return w


    def __RP(self, data, **kwargs):
        '''
        RP split
        Generates some number m of random directions w by sampling
        from the d-dimensional unit sphere, and picks the w which
        maximizes the diameter of projected data from the current node:
        w <- argmax_(w_1, w_2, ..., w_m) max_(x1, x2 in node) |w' (x1 - x2)|
        '''
        k   = kwargs['samples_rp']


        # sample directions from d-dimensional normal
        print(self.__d)
        W   = numpy.random.randn( k, self.__d)

        # normalize each sample to get a sample from unit sphere
        for i in range(k):
            W[i] /= numpy.sqrt(numpy.sum(W[i]**2))
            pass

        # Find the direction that maximally spreads the data:

        min_val = numpy.inf * numpy.ones(k)
        max_val = -numpy.inf * numpy.ones(k)

        for i in self.__indices:
            Wx      = numpy.dot(W, data[i])
            min_val = numpy.minimum(min_val, Wx)
            max_val = numpy.maximum(max_val, Wx)
            pass

        return W[numpy.argmax(max_val - min_val)]

    # end spatialtree class

    class invertedmap(object):

        def __init__(self, T):
            '''
            Construct an inverted map from a spatialtree object T.
            I   = spatialtree.invertedmap(T)
            This provides a more space-efficient data-structure for fast
            retrieval of a static dataset.
            '''
            if not isinstance(T, spatialtree):
                raise TypeError('Argument must be of type: spatialtree')

            # Construct the index
            self.__map      = {}
            self.__leafsets = []
            
            # Leaf-generator helper function
            def leafWalker():
                for node in T.traverse():
                    if node.isLeaf():
                        yield node
                pass

            for (i, node) in enumerate(leafWalker()):
                # Construct a set for the i'th leaf
                leafset = set()

                for item in node:
                    # Add each item contained in the leaf to its set
                    leafset.add(item)

                    # Map the item to the new leaf-set
                    if item not in self.__map:
                        self.__map[item] = set()
                    self.__map[item].add(i)
                    pass

                # Add the new leafset to our list
                self.__leafsets.append(leafset)
                pass
            pass

        def __contains__(self, k):
            '''
            Test if item k is contained in the invertedmap
            '''
            return k in self.__map

        def remove(self, k):
            '''
            Remove an item from the invertedmap
            '''
            if k not in self:
                raise KeyError(k)

            # Remove k from each of its leaf sets
            for s in self.__map[k]:
                self.__leafsets[s].remove(k)
                pass

            # Remove the mapping for k
            del self.__map[k]
            pass

        def __len__(self):
            '''
            Return the number of items in this invertedmap
            '''
            return len(self.__map)

        def numSets(self):
            '''
            Return the number of leaf sets in this invertedmap
            '''
            return len(self.__leafsets)

        def __retrievalSet(self, k):
            '''
            S = invertedmap.__retrievalset(index)
            
            Get the retrieval set for the given item index.
            '''

            RS = set()

            for s in self.__map[k]:
                RS |= self.__leafsets[s]
                pass

            # Remove self from the retrieval set
            RS.remove(k)

            return RS

        def k_nearest(self, data, **kwargs):
            '''
            neighbors = I.k_nearest(data, k=10, index=X)
            data:       the data matrix/dictionary
            k:          the number of (approximate) nearest neighbors to return
            index=X:    the index of the query point
            Returns:
            A sorted list of the indices of k-nearest (approximate) neighbors of the query
            '''
            
            if 'k' not in kwargs:
                raise Exception('k_nearest called with no value of k')

            if not isinstance(kwargs['k'], int):
                raise TypeError('k_nearest must be called with an integer value of k')

            if kwargs['k'] < 1:
                raise ValueError('k must be a positive integer')

            if 'index' in kwargs:
                x = data[kwargs['index']]
            else:
                raise Exception('k_nearest called with no target index')

            # Now compute distance from query point to the retrieval set
            def dg(S):
                for i in S:
                    yield (numpy.sum((x-data[i])**2), i)
                pass

            # Pull out indices in sorted order
            return [i for (d,i) in heapq.nsmallest(kwargs['k'], dg(self.__retrievalSet(kwargs['index'])))]
