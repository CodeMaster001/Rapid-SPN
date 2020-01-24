#!/usr/bin/env python
import sys
import os
sys.setrecursionlimit(1000000000)
import numpy
import scipy.stats
from itertools import islice
import random
import heapq
from spn.algorithms.TransformStructure import Prune,Compress
from spn.algorithms.Validity import is_valid
from spn.structure.Base import Product, Sum, assign_ids, rebuild_scopes_bottom_up
from spn.structure.leaves.parametric.Parametric import create_parametric_leaf
from spn.algorithms.splitting.RDC import get_split_cols_RDC_py, get_split_cols_RDC_py
from collections import deque
from multiprocessing import Process
import numpy as np
from sklearn.cluster import KMeans
import multiprocessing

from spn.structure.Base import Context
from spn.io.Graphics import plot_spn
from spn.algorithms.Sampling import sample_instances
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.splitting.RDC import get_split_cols_RDC_py, get_split_rows_RDC_py
from sklearn.datasets import load_iris,load_digits,fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from spn.algorithms.LearningWrappers import learn_parametric
from spn.gpu.TensorFlow import *
from spn.structure.Base import Product, Sum, assign_ids, rebuild_scopes_bottom_up
from sklearn.metrics import accuracy_score
from numpy.random.mtrand import RandomState
from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier
from spn.algorithms.TransformStructure import Prune,Compress,SPN_Reshape
import urllib
import sys
sys.path.append('.')
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from spn.gpu.TensorFlow import eval_tf
from spn.structure.Base import *
from spn.structure.leaves.cltree import *
import time;
import numpy as np, numpy.random
import sys;
import logging
np.random.seed(42)
import traceback


class NODE_TYPE:
    SUM_NODE= 0;
    PRODUCT_NODE = 1;
    LEAF_NODE=3;
    ROOT=4
    NAIVE=5
#gini index implementation

def gini(data,index):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    data = data[:,index]
    data = data.flatten()
    if np.amin(data) < 0:
        # Values cannot be negative:
        data -= np.amin(data)
    # Values cannot be 0:
    data += 0.0000001
    # Values must be sorted:
    data = np.sort(data)
    # Index per array element:
    index_shape = np.arange(1,data.shape[0]+1)
    # Number of array elements:
    n = data.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index_shape - n  - 1) * data)) / (n * np.sum(data)))




class FriendSPN(object):
#FrienhSPN optimizer and Random Projection

    def __init__(self, data,spn_object=None,ds_context=None,leaves_size=8000,scope=None,prob=0.7,indices=None, height=None,sample_rp=10,TYPE=NODE_TYPE.SUM_NODE,index=-1,default_scope=True):
        self.prob = prob
        self.leaves_size = leaves_size
        self.spn_node = spn_object
        self.scope = scope
        self.ds_context = ds_context
        self.data = data;
        self.height = height;
        self.sample_rp = sample_rp
        self.indices= indices
        self.TYPE=TYPE
        self.index = index;
        if self.scope is None:
            self.scope = list(set(list(range(0,data.shape[1]))))

        #self.split_cols = get_split_cols_RDC_py(ohe=False, threshold=0.2,n_jobs=10)

        if self.indices is None:
            self.indices   = range(len(data))
            pass
        print('we are at height :'+str(height) )

        for x in self.indices:
            self.d = len(data[x])
            break

        for x in self.indices:
            self.__d = len(data[x])
            break


    
    def chunks(self,lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
        
        
    
    
    def update_ids(self):
        assign_ids(self.spn_node)
        rebuild_scopes_bottom_up(self.spn_node)
        #self.spn_node = Prune(self.spn_node)

    def __calculate_gini(self,data,ds_context,scope,threshold=1.0):
        self.scope = np.sort(self.scope)
        temp = np.array(data[:,scope]) #apply existing scope
        temp = np.array(temp)
        print('called')
        split_cols = list()
        print('Current scope:'+str(self.scope))
        gini_values = np.zeros(shape=(temp.shape[1],temp.shape[1]))
        for i in range(0,temp.shape[1]):
            process = list()
            for j in range(0,temp.shape[1]):
                if i==j:
                    continue;
                else:
                    gini_values[i,j] =scipy.spatial.distance.dice(temp[:,i],temp[:,j])

        cands = self.build_candidates(gini_values,4)
        print(cands)
        print('------')
        scopes=self.optimize_scope(temp,self.ds_context,cands)
        print('Selected_scopes')
        print(scopes)

        return scopes;
     
        '''
        print('passed')
        average_value=np.mean(gini_values,axis=0)
 
        index  = numpy.argmax(average_value)
        print(average_value)
        left=list()
        right=list();
        for i in range(0,len(average_value)):
            if i==index:
                left.append(i)
                continue;
            if np.abs(average_value[i]-average_value[index])<threshold :
                left.append(i)
            else:
                right.append(i)

        if len(left)>0:
            split_cols.append(left)
        if len(right)>0:
            split_cols.append(right)
        print('total slices:'+str(split_cols))
        print("gini complete")
        return split_cols
        '''

    #perform random projection on scope


     
    def project(self,data,scope,sample_rp,**kwargs):
        self.w = self.__RP(data,scope,sample_rp,scope_limit=4,**kwargs)

        temp = data[:,scope]
        # Project onto split direction
        wx = {}
        for i in self.indices:
            wx[i] = numpy.dot(self.w, temp[i])
            pass

        threshold = np.mean(list(wx.values())) #center point of threshold

        # Partition the data
        left_set    = set()
        right_set   = set()

        right_data = list();
        left_data = list();



        for i in self.indices:
            if wx[i] >= threshold:
                right_set.add(i)
            else:
                left_set.add(i)

        del wx  # Don't need scores anymore

        total = len(left_set) + len(right_set)

        return left_set,len(left_set)/total,right_set,len(right_set)/total,threshold
    
    def __RP(self,data,scope,sample_rp,scope_limit,**kwargs):

        '''
        RP split
        Generates some number m of random directions w by sampling
        from the d-dimensional unit sphere, and picks the w which
        maximizes the diameter of projected data from the current node:
        w <- argmax_(w_1, w_2, ..., w_m) max_(x1, x2 in node) |w' (x1 - x2)|

        '''
        print(scope)
        start = 0;
        final_w = list();
        temp = data[:,scope]

        for x in self.indices:
            self.d = len(temp[x])
            break
        # sample directions from d-dimensional normal
        W   = numpy.random.randn(sample_rp, self.d )
        
        # normalize each sample to get a sample from unit sphere
        for i in range(sample_rp):
            W[i] /= numpy.sqrt(numpy.sum(W[i]**2))
            pass

        # Find the direction that maximally spreads the data:

        min_val = numpy.inf * numpy.ones(sample_rp)
        max_val = -numpy.inf * numpy.ones(sample_rp)

        Wx      = numpy.dot(W, temp[i])
        min_val = numpy.minimum(min_val, Wx)
        max_val = numpy.maximum(max_val, Wx)
        temp = W[numpy.argmax(max_val - min_val)]
        return temp;

    def getIndices(self):
        return self.indices;


    def build_candidates(self,features_set,n):
        candidates = list();

        temp = np.array(features_set)
        mean_array= np.mean(temp,axis=1)
        mean_index = int(len(mean_array)/2.0)
 
        if n>=len(mean_array):
            n=len(mean_array)

        for j in range(0,features_set.shape[1]):
            selected_feature = temp[j,:]
            sorted_feature_index = np.argsort(selected_feature)
            sorted_feature_index = [self.scope[i] for i in sorted_feature_index]
            for chunk_index in [1]:
                if chunk_index<=len(sorted_feature_index):
                    sorted_feature_index_temp = list(self.chunks(sorted_feature_index,chunk_index))
                    sorted_feature_index_temp = [i for i in sorted_feature_index_temp if len(i)>=1]
                    candidates.append(sorted_feature_index_temp)
                print(self.scope)
                print(candidates)
        return candidates

    def default_scope(self,data,ds_context):

        s=Sum();
        s.children.append(self.naive_factorization_naive(data=data,scope=self.scope))
        s.weights.append(1.0)
        s.scope.extend(self.scope)
        s=assign_ids(s)
        s=rebuild_scopes_bottom_up(s)
        value=log_likelihood(s,data)
        return np.mean(value)

    def optimize_scope(self,data,ds_context,candidates):
        sorted_scope = np.sort(self.scope)
        print(self.data.shape)
        max_list=list();
        best_cand=-1000000000
        cand_select=[self.scope]
        print('length of candidates:'+str(len(candidates)))
        print(candidates)
        print(data.shape)
        for cand in candidates:
            try:
                s=Sum();
                for child in cand:
                    print(len(cand))
                    print('------')
                    print(self.scope)
                    print(child)
                    s.children.append(self.naive_factorization_naive(data=self.data,scope=child))
                    s.weights.append(1.0/float(len(cand)))

                s=assign_ids(s)
                s=rebuild_scopes_bottom_up(s)
                print(self.data.shape)
                value=np.mean(log_likelihood(s,self.data[:,self.scope]))

                if best_cand<value:
                    best_cand = value
                    cand_select=cand
            except:  
                traceback.print_exc();
                sys.exit(-1)
        print('completed iteration')
        return cand_select


        

        return max_list
    def split_cols(self, data,ds_context,scope,n=2):


        #return self.split_cols(data, self.ds_context, self.scope)

    
        cols_split = self.__calculate_gini(data,ds_context=self.ds_context,scope=scope) #split cols apply scope and gin
        print('Product Node called')
      
        """Yield successive n-sized chunks from l"""
        for i in range(0, len(cols_split)):
            yield data[:,cols_split[i]],cols_split[i]
        
    #node production based on SPN

    def build_leaf_node(self,data,scope,ds_context):
        print('Leaf Node called')
        node = create_parametric_leaf(data[list(self.indices),:].reshape(-1,1), self.ds_context, scope)
        return node;


    def split(self,**kwargs):
    # Store bookkeeping information
        #base cases
        print('At split')
        print('Height:'+str(self.height)+','+'Leaves:'+str(self.leaves_size))

        if self.height<=0 or len(self.indices)< self.leaves_size:
            print('base case')
            node =self.naive_factorization(self.data,self.scope)
            self.spn_node.children[self.index]=node
            return;

        elif len(self.scope)==1 or self.TYPE==NODE_TYPE.LEAF_NODE:
            self.spn_node.children[self.index]=self.build_leaf_node(self.data,self.scope,self.ds_context)
            return;

        elif self.TYPE == NODE_TYPE.SUM_NODE:
            self.build_sum_node(**kwargs)


        elif self.TYPE == NODE_TYPE.PRODUCT_NODE:
            self.build_product_node(**kwargs)

        elif self.TYPE==NODE_TYPE.NAIVE:
            self.spn_node.children[self.index]=self. naive_factorization(self.data,self.scope)

    def build_sum_node(self,**kwargs):
        print('Sum Node  called')

        left_set,left_weight,right_set,right_weight,threshold = self.project(self.data,self.scope,self.sample_rp,**kwargs)

        self.children     = list()

        sum_node = Sum();
        sum_node.scope.extend(self.scope)
        sum_node.weights.append(left_weight)
        sum_node.children.append(None)
        
        if len(left_set)<10:
            node_left = FriendSPN(data=self.data,indices=left_set,spn_object=sum_node,scope=self.scope,ds_context=self.ds_context,height=self.height-1,prob=self.prob,sample_rp=self.sample_rp,TYPE=NODE_TYPE.PRODUCT_NODE,leaves_size=self.leaves_size,index=0)
            pass;
        else:
            node_left = FriendSPN(data=self.data,indices=left_set,spn_object=sum_node,scope=self.scope,ds_context=self.ds_context,height=self.height-1,prob=self.prob,sample_rp=self.sample_rp,TYPE=NODE_TYPE.PRODUCT_NODE,leaves_size=self.leaves_size,index=0)
        
        sum_node.weights.append(right_weight)
        sum_node.children.append(None)

        if len(right_set)<10:
            node_right = FriendSPN(data=self.data,indices=right_set,spn_object=sum_node,scope=self.scope,ds_context=self.ds_context,height=self.height-1,prob=self.prob,sample_rp=self.sample_rp,TYPE=NODE_TYPE.PRODUCT_NODE,leaves_size=self.leaves_size,index=1)
            pass;
        else:
            node_right = FriendSPN(data=self.data,indices=right_set,spn_object=sum_node,scope=self.scope,ds_context=self.ds_context,height=self.height-1,prob=self.prob,sample_rp=self.sample_rp,TYPE=NODE_TYPE.PRODUCT_NODE,leaves_size=self.leaves_size,index=1)


        self.children =[node_left,node_right]
        
        SPNRPBuilder.tasks.append([node_left,kwargs])
        SPNRPBuilder.tasks.append([node_right,kwargs])

        if self.spn_node == None:
            self.spn_node = sum_node;
        else:
            self.spn_node.children[self.index]=sum_node;
        print('complete')

    def build_product_node(self,**kwargs):
        print('product node called')
        scope_list = list()
        '''
        if len(self.indices)<10 or len(self.scope)<15:
            node = FriendSPN(data=self.data,indices=self.indices,spn_object=self.spn_node,scope=self.scope,ds_context=self.ds_context,height=self.height,prob=self.prob,sample_rp=self.sample_rp,TYPE=NODE_TYPE.NAIVE,index=self.index)
            SPNRPBuilder.tasks.append([node,kwargs])
            return;
        '''
        #else:
        node= Product();
        node.scope.extend(self.scope)

        
        rptree = list()
        child_count = -1;
        temp = self.data[list(self.indices)]
    
        if self.spn_node == None:

            self.spn_node = node;
            self.spn_node.scope.extend(self.scope)
        else:
            self.spn_node.children[self.index]=node;

        try:

            for _,scope_slice in self.split_cols(data=self.data[list(self.indices)], ds_context=self.ds_context, scope=self.scope):
                print(scope_slice)
                if len(scope_slice) == 1 and len(temp) !=0:
                    node.scope.extend(scope_slice)
                    child_count = child_count + 1;
                    node.children.append(None)
                    children_friend =FriendSPN(data=self.data,spn_object=node,ds_context=self.ds_context,leaves_size=self.leaves_size,scope=scope_slice,prob=self.prob,indices=self.indices,height=self.height,sample_rp=self.sample_rp,TYPE=NODE_TYPE.LEAF_NODE,index=child_count)
                    SPNRPBuilder.tasks.append([children_friend,kwargs])
                
                elif len(scope_slice)>1:
                    node.scope.extend(scope_slice)
                    child_count = child_count + 1;
                    node.children.append(None)
                    children_friend =FriendSPN(data=self.data,spn_object=node,ds_context=self.ds_context,leaves_size=self.leaves_size,scope=scope_slice,prob=self.prob,indices=self.indices,height=self.height,sample_rp=self.sample_rp,TYPE=NODE_TYPE.SUM_NODE,index=child_count)
                    SPNRPBuilder.tasks.append([children_friend,kwargs])
                '''
                elif len(temp) <20:
                    node.scope.extend(scope_slice)
                    child_count = child_count + 1;
                    node.children.append(None)
                    children_friend =FriendSPN(data=self.data,spn_object=node,ds_context=self.ds_context,leaves_size=self.leaves_size,scope=scope_slice,prob=self.prob,indices=self.indices,height=self.height,sample_rp=self.sample_rp,TYPE=NODE_TYPE.NAIVE,index=child_count)
                  SPNRPBuilder.tasks.append([children_friend,kwargs])
                '''
        except:
            traceback.print_exc()
            node.scope.extend(self.scope)
            child_count = child_count + 1;
            node.children.append(None)
            children_friend =FriendSPN(data=self.data,spn_object=node,ds_context=self.ds_context,leaves_size=self.leaves_size,scope=self.scope,prob=self.prob,indices=self.indices,height=self.height,sample_rp=self.sample_rp,TYPE=NODE_TYPE.SUM_NODE,index=child_count)
            SPNRPBuilder.tasks.append([children_friend,kwargs])
        
    
        
        #self.spn_node.children[self.index]=node;
        #children_friend =FriendSPN(data=self.data,spn_object=node,ds_context=self.ds_context,leaves_size=self.leaves_size,scope=self.scope,prob=self.prob,indices=self.indices,height=self.height,sample_rp=self.sample_rp,TYPE=NODE_TYPE.SUM_NODE,index=0)

        #SPNRPBuilder.tasks.append([children_friend,kwargs])


    def naive_factorization(self, data,scope,is_indices=True):
        print('At leaves:'+str(data.shape))

        spn_node = Product()
        spn_node.scope.extend(scope)
        scope = list(set(scope))
        indices = self.getIndices()
        if is_indices==True:
            node_info = data[list(indices),:]
        else:
            node_info=data
        for i in range(0,len(scope)):
            node = create_parametric_leaf(node_info[:,i].reshape(-1,1), self.ds_context, [scope[i]])
            spn_node.children.append(node)
        return spn_node

    def naive_factorization_naive(self, data,scope,is_indices=True):
        print('At leaves:'+str(data.shape))
        spn_node = Product()
        spn_node.scope.extend(scope)
        scope = list(set(scope))
        node_info=data
        for i in range(0,len(scope)):
            node = create_parametric_leaf(node_info[:,i].reshape(-1,1), self.ds_context, [scope[i]])
            spn_node.children.append(node)
        return spn_node

    def bfs(root, func):
        seen, queue = set([root]), collections.deque([root])
        while queue:
            node = queue.popleft()
            func(node)
            if not isinstance(node, Leaf):
                for c in node.children:
                    if c not in seen:
                        seen.add(c)
                        queue.append(c)

def same_prob(node):
    if isinstance(node,Sum):
        node.weights= np.random.dirichlet(np.ones(len(node.weights)),size=1)[0]

class SPNRPBuilder(object):
    
    tasks =list()

    def __init__(self, data,spn_object=None,ds_context=None,leaves_size=8000,scope=None,threshold=0.4,prob=0.7,indices=None,height=None,sample_rp=10,**kwargs):
        print('Intialzied for height' + str(height))
        self.root= FriendSPN(data=data,spn_object=spn_object,ds_context=ds_context,leaves_size=leaves_size,scope=scope,prob=prob,indices=indices,height=height,sample_rp=sample_rp,TYPE=NODE_TYPE.SUM_NODE)
        SPNRPBuilder.tasks.append([self.root,kwargs])
        self.data=data;
 

    def build_spn(self):
        while SPNRPBuilder.tasks:
            sp,kwargs =  SPNRPBuilder.tasks.pop();
            sp.split(**kwargs)
            #print(kwargs)
        return self.after()

    def after(self):
        spn_node = self.root.spn_node;
        assign_ids(spn_node)
        rebuild_scopes_bottom_up(spn_node)
        import sys
        spn_node = Prune(spn_node)
        print("exited")
        print(spn_node)
        self.root.spn_node=spn_node #apply prune
        return self.root;


