#!flask/bin/python

from __future__ import division
from flask import Flask, jsonify, abort, request, make_response, url_for
import json
import pickle
import base64
import numpy
import math
import scipy
from copy import deepcopy
from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA
from sklearn import linear_model
from numpy  import array, shape, where, in1d
import ast
import threading
import Queue
import time
import random
import sklearn
from random import randrange
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

app = Flask(__name__, static_url_path = "")


def getJsonContentsTrain (jsonInput):
    try:
        dataset = jsonInput["dataset"]
        predictionFeature = jsonInput["predictionFeature"]
        parameters = jsonInput["parameters"]
        print parameters
        datasetURI = dataset.get("datasetURI", None)
        dataEntry = dataset.get("dataEntry", None)

        variables = dataEntry[0]["values"].keys() 
        variables.sort() #### 24062015
        datapoints =[]
        target_variable_values = []
        for i in range(len(dataEntry)):
            datapoints.append([])

        for i in range(len(dataEntry)):
            for j in variables:		
                if j == predictionFeature:
                    target_variable_values.append(dataEntry[i]["values"].get(j))
                else:
                    datapoints[i].append(dataEntry[i]["values"].get(j))				

        variables.remove(predictionFeature)		
        print variables
    except(ValueError, KeyError, TypeError):
        print "Error: Please check JSON syntax... \n"
	
    return variables, datapoints, predictionFeature, target_variable_values, parameters

def getJsonContentsTest (jsonInput):
    try:
        dataset = jsonInput["dataset"]	
        rawModel = jsonInput["rawModel"]
        additionalInfo = jsonInput["additionalInfo"]
		
        datasetURI = dataset.get("datasetURI", None)
        dataEntry = dataset.get("dataEntry", None)
		
        predictionFeature = additionalInfo[0].get("predictedFeature", None)
		
        variables = dataEntry[0]["values"].keys() 
        variables.sort() #### 24062015

        datapoints =[]
        for i in range(len(dataEntry)):
            datapoints.append([])
			
        for i in range(len(dataEntry)):
            for j in variables:		
                datapoints[i].append(dataEntry[i]["values"].get(j))				
		
    except(ValueError, KeyError, TypeError):
        print "Error: Please check JSON syntax... \n"
		
    return variables, datapoints, predictionFeature, rawModel

def entropy(data, attribute):
    value_frequencies = {}
    data_entropy = 0.0

    for record in data:
        if (value_frequencies.has_key(record[attribute])):
            value_frequencies[record[attribute]] += 1.0  
        else:
            value_frequencies[record[attribute]] = 1.0
			
    for frequency in value_frequencies.values():
        data_entropy += (-frequency/len(data)) * math.log(frequency/len(data), 2)   
    
    return data_entropy

def information_gain(data, attribute, target_attribute):

    value_frequencies = {}
    subset_entropy = 0.0

    for record in data:
        if (value_frequencies.has_key(record[attribute])):
            value_frequencies[record[attribute]] += 1.0
        else:
            value_frequencies[record[attribute]] = 1.0
			
    for value in value_frequencies.keys():
        value_prior = value_frequencies[value] / sum(value_frequencies.values())    
        data_subset = [record for record in data if record[attribute] == value]
        subset_entropy += value_prior * entropy(data_subset, target_attribute)
	
    information = entropy(data, target_attribute) - subset_entropy
	
    return information  

def unique(lst):
    lst = lst[:]
    unique_lst = []

    for item in lst:
        if unique_lst.count(item) <= 0:
            unique_lst.append(item)
            
    return unique_lst

def get_values(data, attr):
    data = data[:]
    return unique([record[attr] for record in data])

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def majority_value(data, target_attr):
    data = data[:]
    return most_frequent([record[target_attr] for record in data])

def most_frequent(lst):
    lst = lst[:]
    highest_freq = 0
    most_freq = None

    for val in unique(lst):
        if lst.count(val) > highest_freq:
            most_freq = val
            highest_freq = lst.count(val)
            
    return most_freq	

def get_examples(data, attr, value):
    data = data[:]
    rtn_lst = []
    
    if not data:
        return rtn_lst
    else:
        for record in data:
            if record[attr] == value:
                rtn_lst.append(record)    
        return rtn_lst

def numerical_information_gain(data, attribute, target_attribute):
    value_frequencies = {}
    subset_entropy = 0.0
	
    for record in data:
        if (value_frequencies.has_key(record[attribute])):
            value_frequencies[record[attribute]] += 1.0
        else:
            value_frequencies[record[attribute]] = 1.0
    
    
    all_numeric = get_values(data, attribute)
    all_numeric = [float(num) for num in all_numeric]
    all_numeric.sort()

    best_gain = 0.0
    best_split_number = 0.0

    for numeric_index in range (1,len(all_numeric)-1): 
        temp = deepcopy(data)
        for record in temp:
            if float(record[attribute]) >= all_numeric[numeric_index]:
                record[attribute] = ">="+str(all_numeric[numeric_index])
            else:
                record[attribute] = "<"+str(all_numeric[numeric_index])

        gain = information_gain(temp, attribute, target_attribute)

        if (gain >= best_gain and attribute != target_attribute):
            best_gain = gain
            best_split_number = all_numeric[numeric_index]	

    temp = deepcopy(data)		
    for record in temp:
        if float(record[attribute]) >= best_split_number:
            record[attribute] = ">="+str(best_split_number)
        else:
            record[attribute] = "<"+str(best_split_number)
    return [temp, best_split_number]

def mutual_information(data ,x_index, y_index, logBase, debug = False): 

    summation = 0.0

    values_x = set([data[i].get(x_index) for i in range (len(data))])
    values_lx = list([data[i].get(x_index) for i in range (len(data))])
        
    values_y = set([data[i].get(y_index) for i in range (len(data))])
    values_ly = list([data[i].get(y_index) for i in range (len(data))])
        
    for value_x in values_x:
        for value_y in values_y:
            px = values_lx.count(value_x) / len(data)
            py = values_ly.count(value_y) / len(data)
            
            indexesX = [i for i,x in enumerate(values_lx) if x == value_x]
            indexesY = [i for i,y in enumerate(values_ly) if y == value_y]

            pxy = len(where(in1d(indexesX, indexesY)==True)[0] ) / len(data) 

            if pxy > 0.0:
                summation += pxy * math.log((pxy / (px*py)), logBase)

    return summation

def decision_tree(data, attributes, target_attribute, algorithm, logBase = 10, attributes_used = []):

    class_values = [record[target_attribute] for record in data]
    default = majority_value(data, target_attribute)

    if not data or (len(attributes) - 1) <= 0:
        return default
    elif class_values.count(class_values[0]) == len(class_values):
        return class_values[0]
    else:
        if algorithm == "id3":
            best_gain = 0.0
            best_attr = None
			
            for attribute in attributes:
                unique_datapoints = []

                unique_datapoints = [record[attribute] for record in data]
				
                temp = []	
                best_split_number = 0.0				

                if len(set(unique_datapoints)) > 2:
                    temp, best_split_number = numerical_information_gain(data, attribute, target_attribute) 
                    data = deepcopy(temp)
                gain = information_gain(data, attribute, target_attribute)
				
                if (gain >= best_gain and attribute != target_attribute):
                    best_gain = gain
                    best_attr = attribute
         
            tree = {best_attr:{}}
            
            unique = [record[best_attr] for record in data]
            unique = list(set(unique)) 
			
            for value in unique: 
                subtree = decision_tree( get_examples(data, best_attr, value), [attr for attr in attributes if attr != best_attr], target_attribute, algorithm)
                tree[best_attr][value] = subtree         
                if best_attr not in attributes_used:
                    attributes_used.append(best_attr)

        if algorithm == "mci":
            best_gain = 0.0
            best_attr = None
			
            for attribute in attributes:
                unique_datapoints = []

                unique_datapoints = [record[attribute] for record in data]
			
                temp = []	
                best_split_number = 0.0				
				
                if len(set(unique_datapoints)) > 2:
                    temp, best_split_number = numerical_information_gain(data, attribute, target_attribute) 
                    data = deepcopy(temp)

                gain = mutual_information(data ,attribute, target_attribute, logBase, debug = False)

                if (gain >= best_gain and attribute != target_attribute):
                    best_gain = gain
                    best_attr = attribute           
            tree = {best_attr:{}}
            
            unique = [record[best_attr] for record in data]
            unique = list(set(unique)) 

            for value in unique: 
                subtree = decision_tree( get_examples(data, best_attr, value), [attr for attr in attributes if attr != best_attr], target_attribute, algorithm, logBase)
                tree[best_attr][value] = subtree 
                if best_attr not in attributes_used:
                    attributes_used.append(best_attr)
    
    return tree, attributes_used

def dicreader (dic, attributes, data_instance):
    if isinstance(dic, dict):
        for k,v in dic.items():
            if k in attributes:
                for checkVal in v.keys(): 
                    if float(checkVal.strip(">=")) or float(checkVal.strip("<")):
                        if (">=" in str(checkVal)) and (float(test_value) >= float(checkVal.strip(">="))):
                            test_value = checkVal
                        elif (">=" in str(checkVal)) and (test_value < float(checkVal.strip("<"))):
                            test_value = checkVal
                    else:
                        test_value = data_instance[k]
                return dicreader(v[test_value], attributes, data_instance)
    else:
        return dic
	
def tree_predict(variables, datapoints, predictionFeature, rawModel):
    decoded = base64.b64decode(rawModel)
    modeldata = ast.literal_eval(decoded)
    predictionList = []

    for data_instance in datapoints:
        pred = dicreader (modeldata, variables, data_instance)
        finalPrediction = {predictionFeature:pred} 
        predictionList.append(finalPrediction)

    return predictionList


# begin threading
""" 
def best_latent_variable(X, Y, latent_variables, num_instances):
    r2_best = -10000
    lv_best = 1 
	
    for lv in range (1, latent_variables): 
        r2_cumulative = 0 
        myThreads = []
        for instance in range (0, num_instances):
            myThreads.append(enthread( target = crossval, args = (X,Y,instance,lv)) )
        for i in range (0,num_instances):
            r2 = myThreads[i].get()
            #r2_cumulative += myThreads[i].get()
            if isinstance(r2, str):
                r2_cumulative += r2
            elif isinstance(r2[0], str):
                r2_cumulative += r2[0]
            else: 
                r2_cumulative += r2[0][0]
            
  
        if (r2_cumulative/num_instances) > r2_best:
            r2_best = (r2_cumulative/num_instances)
            lv_best = lv
    return lv_best

def crossval (X,Y,instance,lv):
    meanY4r2 = numpy.mean(Y)

    X_train = deepcopy(X)
    X_test = [X_train.pop(instance)]
			
    Y_train = deepcopy(Y)
    Y_test = [Y_train.pop(instance)]   	
			
    plsca = PLSRegression(n_components=lv)
			
    plsca.fit(X_train, Y_train)
			
    predY = plsca.predict(X_test) ####
    RSS = numpy.power (Y_test[0] - predY, 2)
    TSS = numpy.power (Y_test[0] - meanY4r2, 2)
    r2 = 1 - (RSS/TSS)
    return r2
""" 
# end threading

# begin working version 23062015
"""
def best_latent_variable(X, Y, latent_variables, num_instances):
    r2_best = -10000
    lv_best = 1 
	
    for lv in range (1, latent_variables): 
        r2_cumulative = 0 #23062015
        for instance in range (0, num_instances):

            meanY4r2 = numpy.mean(Y)

            X_train = deepcopy(X)
            X_test = [X_train.pop(instance)]
			
            Y_train = deepcopy(Y)
            Y_test = [Y_train.pop(instance)]   	
			
            plsca = PLSRegression(n_components=lv)
			
            plsca.fit(X_train, Y_train)
			
            #X_train_r, Y_train_r = plsca.transform(X_train, Y_train)
            #X_test_r, Y_test_r = plsca.transform(X_test, Y_test)
            
            predY = plsca.predict(X_test) ####
            RSS = numpy.power (Y_test[0] - predY, 2)
            TSS = numpy.power (Y_test[0] - meanY4r2, 2)
            r2 = 1 - (RSS/TSS)
            #r2 = plsca.score(X_test, Y_test)   

            #### 23062015 local optimum
            if isinstance(r2, str):
                r2_cumulative += r2
            elif isinstance(r2[0], str):
                r2_cumulative += r2[0]
            else: 
                r2_cumulative += r2[0][0]
        #print "lv: ", lv, "r2: ",(r2_cumulative/num_instances),"r2 best: ", r2_best, "lv_best: ", lv_best      
        if (r2_cumulative/num_instances) > r2_best:
            r2_best = (r2_cumulative/num_instances)
            lv_best = lv
        #else:
        #    return lv_best #### 23062015 local optimum
    #print "R2", r2_best, "LV", lv_best ###########
    return lv_best
"""
# end working version 23062015

#BEGIN PLS
def getR2 (Y, predY):
    R2 = sklearn.metrics.r2_score(Y, predY)
    """
    meanY4r2 = numpy.mean(Y)
    meanYpred4r2 = numpy.mean(predY)

    SSXX = 0
    SSYY = 0
    SSXY = 0
    for i in range (len(Y)):
        SSXX += numpy.power ((Y[i] - meanY4r2), 2)
        SSYY += numpy.power ((predY[i] - meanYpred4r2), 2) 
        SSXY += (Y[i] - meanY4r2)*(predY[i] - meanYpred4r2)
    
    if SSXX ==0 or SSYY ==0:
        R2wolfram = 0
    else:
        R2wolfram = numpy.power(SSXY, 2)/(SSXX*SSYY)

    return R2wolfram
    """
    return R2

def best_latent_variable(X, Y, latent_variables, num_instances):
    r2_best = -10000
    lv_best = 1 

    for lv in range (1, latent_variables): 
        r2_cumulative = 0 #23062015
        
        myIndices = range (0, num_instances)

        plsca = PLSRegression(n_components=lv,scale=False)
        #1
        #scores = cross_validation.cross_val_score( plsca, X, Y, cv=10)
        #scores = cross_validation.cross_val_score( plsca, X, Y, cv=num_instances) # LOO
        #r2 = numpy.mean(scores)

        #2
        predY = cross_validation.cross_val_predict( plsca, numpy.array(X), numpy.array(Y), cv=10)
        r2 = getR2 (Y, predY)

        if (r2 > r2_best):
            r2_best = r2
            lv_best = lv

    return lv_best

def get_vip (fin_pls, lv_best, current_attribute, attributes_gone, attributes):
    
    firstSum = 0
    secondSum = 0

    SS = numpy.power(fin_pls.y_loadings_[0][lv_best-1],2) * numpy.matrix.transpose(fin_pls.x_scores_[lv_best-1]) * fin_pls.x_scores_[lv_best-1]

    colSums = 0
		
    for i in range (0, attributes): 
        colSums += numpy.power( fin_pls.x_weights_[i][lv_best-1], 2 )			
    if colSums ==0:
        Wdiv = 0
    else: 
        Wdiv = fin_pls.x_weights_[current_attribute][lv_best-1] / math.sqrt(colSums)

    firstSum += SS*Wdiv
    secondSum += SS
    VIPcurrent = firstSum*attributes / secondSum
	
    if not isinstance(VIPcurrent, str):
        return VIPcurrent[0]
    else:
        return VIPcurrent
		
#################################a
def enthread(target, args):
    q = Queue.Queue()
    def wrapper():
        q.put(target(*args))
    t = threading.Thread(target=wrapper)
    t.start()
    return q
#################################a

def plsvip (X, Y, V, lat_var):
    attributes = len(X[0])

    if not lat_var:
        latent_variables = attributes
    else:
        latent_variables = lat_var
		
    num_instances = len(X)	
	
    attributes_gone = []

    min_att = -1	

    #start_time = time.time()
    #attr_time = time.time()
    #time_counter = 0
    while attributes>0: 
        #if (attributes +9) %10 ==0:
        #    print "total time: ", time.time() - start_time
        #    print "attr time: ", time.time() - attr_time
        #    attr_time = time.time()

        if (latent_variables == 0) or (latent_variables > attributes):	
            latent_variables = attributes	

        lv_best = best_latent_variable(X, Y, latent_variables, num_instances)
        #print "current best lv: ", lv_best, "num. attr. ", attributes ####

        #fin_pls = PLSCanonical(n_components = lv_best)
        fin_pls = PLSRegression(n_components = lv_best,scale=False)

        ## previous
        #fin_pls.fit(X, Y)
        #currentR2 = fin_pls.score(X, Y)  
        fin_pls.fit(X, Y)
        predY = fin_pls.predict(X, Y)


        #print Y[0], predY[0]
        currentR2 = getR2 (Y, predY)
        #print "R2 ", currentR2, "Avg", numpy.mean(Y), "Pred", numpy.mean(predY), "Attr", attributes, "Lat", lv_best

        min_vip = 1000

        #print lv_best, attributes, currentR2
        if min_att ==-1:
            attributes_gone.append(["None", currentR2, attributes, lv_best])

        ##########################################r
        #threaded version
        """ 
        myThreads = []
        VIPcurrent = []
        for i in range (0,attributes):
            myThreads.append(enthread( target = get_vip, args = (fin_pls, lv_best, i, attributes_gone, attributes  )) )
        for i in range (0,attributes):
            VIPcurrent.append(myThreads[i].get())
      
        min_vip = min(VIPcurrent)
        min_att = VIPcurrent.index(min_vip)
        """ 
        # Working version
        #"""
        for i in range (0,attributes):
            VIPcurrent = get_vip (fin_pls, lv_best, i, attributes_gone, attributes  )
            if VIPcurrent< min_vip:
                min_vip = VIPcurrent
                min_att = i
        #"""
        ##########################################r
        if min_att >-1:
            attributes_gone.append([V[min_att], round(currentR2,2), attributes, lv_best]) ####### CURRENT : to BE popped, NOT already popped
        V.pop(min_att)

        for i in range (num_instances):
            X[i].pop(min_att)

        attributes -= 1		
    #print attributes_gone ####
    #time_counter +=1
    return attributes_gone

def bestpls(vipMatrix, X, Y, V):

    ###########################
    #bestR2 = -10000
    #lv_best = 1
    #position = 1
    ###########################
    bestR2 = vipMatrix[0][1]
    lv_best = vipMatrix[0][3]
    position = 0

    for entries in range (len(vipMatrix)):

        if vipMatrix[entries][1] > bestR2:   
            position = entries
            bestR2 = vipMatrix[entries][1]
            lv_best = vipMatrix[entries][3]

    #################################################################################################qq
    variables = []    
    for i in range (1, position): # not position + 1, as the vipMatrix[position] holds the next variable to be removed
        variables.append(vipMatrix[i][0])

    V_new_Indices = []
    for i in variables: # removed variable names in random order
        V_new_Indices.append(V.index(i))

    #if V == sorted(V):
    #    print "\nV ok!\n"

    # keep names == separate
    V_new = deepcopy(V)
    for i in variables:
        V_new.remove(i)
        
    X_new = []
    for i in range (len(X)):
        X_new.append([])

    variables_sent = [] ####
    for i in range (len(X)):
        for j in range (len(V)):
            if j not in V_new_Indices:
                #if V[j] not in variables_sent: ####
                #    variables_sent.append(V[j])####
                X_new[i].append(X[i][j])

    # epic test
    if not V_new == sorted(V_new):
        return base64.b64encode("nein"), [], [], 0

    #validity tests
    #for i in range (len (variables_sent)):
    #    if variables_sent[i] == V_new[i]:
    #        print "ok", i
    #print "var: ", len(V), "selected: ", len(V_new), "data (var) init length: ", len(X[0]), "data (var) now length: ", len(X_new[0])
    """ 
    # PREVIOUS
    variables = []    
    for i in range (1, position):
        variables.append(vipMatrix[i][0])

    V_new = deepcopy(V)
    for i in variables:
        V_new.remove(i) ################ remove by index??? CHECK!!!!

    X_new = []
    for i in range (len(X)):
        X_new.append([])

    for i in range (len(X)):
        for j in range (len(V_new)): ####### HERE ALSO
            X_new[i].append(X[i][j])
    """
    #################################################################################################qq

    best_pls = PLSRegression(n_components = lv_best,scale=False)
    best_pls.fit(X_new, Y)

    ## debug 
    # predY = best_pls.predict(X_new, Y)
    #print "final R2", getR2(Y,predY)
    #for i in range (len(Y)):
    #    print Y[i], predY[i]
    ## end debug

    saveas = pickle.dumps(best_pls)
    encoded = base64.b64encode(saveas)	
	
    return encoded, X_new, V_new, lv_best


def plsvip_predict (variables, datapoints, predictionFeature, rawModel):
    decoded = base64.b64decode(rawModel)
    pls_vip = pickle.loads(decoded)
    predictionList = []

    for i in range (len(datapoints)):
        temp = pls_vip.predict(datapoints[i])
        #print datapoints[i]
        ###if isinstance(temp[0], list):
        if isinstance(temp[0], (list, tuple, numpy.ndarray)):
            finalPrediction = {predictionFeature:temp[0][0]} 
        else:
            finalPrediction = {predictionFeature:temp[0]} 
        predictionList.append(finalPrediction)
    return predictionList

############END PLS

############BEGIN LM
def lm (variable_values, target_variable_values):

    clf = linear_model.LinearRegression()
	
    clf.fit (variable_values, target_variable_values) 
    saveas = pickle.dumps(clf)

    encoded = base64.b64encode(saveas)	
    return encoded

def lm_test (variables, datapoints, predictionFeature, rawModel):

    decoded = base64.b64decode(rawModel)
    clf2 = pickle.loads(decoded)
    predictionList = []
    for i in range (len(datapoints)):
        temp = clf2.predict(datapoints[i])
        finalPrediction = {predictionFeature:temp[0]}
        predictionList.append(finalPrediction)
    return predictionList

############END LM

############BEGIN GNB
def gnb (variable_values, target_variable_values):

    gnb = GaussianNB()
	
    gnb.fit (variable_values, target_variable_values) 
    saveas = pickle.dumps(gnb)

    encoded = base64.b64encode(saveas)	
    return encoded

def gnb_test (variables, datapoints, predictionFeature, rawModel):

    decoded = base64.b64decode(rawModel)
    gnb2 = pickle.loads(decoded)
    predictionList = []
    for i in range (len(datapoints)):
        temp = gnb2.predict(datapoints[i])
        if isinstance (temp,str):
            finalPrediction = {predictionFeature:temp}
        else:
            #here !!
            finalPrediction = {predictionFeature:temp[0]}
        predictionList.append(finalPrediction)
    return predictionList

############END GNB

############BEGIN MNB
def mnb (variable_values, target_variable_values):

    mnb = GaussianNB()
	
    mnb.fit (variable_values, target_variable_values) 
    saveas = pickle.dumps(mnb)

    encoded = base64.b64encode(saveas)	
    return encoded

def mnb_test (variables, datapoints, predictionFeature, rawModel):

    decoded = base64.b64decode(rawModel)
    mnb2 = pickle.loads(decoded)
    predictionList = []
    for i in range (len(datapoints)):
        temp = mnb2.predict(datapoints[i])
        if isinstance (temp,str):
            finalPrediction = {predictionFeature:temp}
        else:
            #here !!
            finalPrediction = {predictionFeature:temp[0]}
        predictionList.append(finalPrediction)
    return predictionList

############END MNB

############BEGIN BNB
def bnb (variable_values, target_variable_values):

    bnb = GaussianNB()
	
    bnb.fit (variable_values, target_variable_values) 
    saveas = pickle.dumps(bnb)

    encoded = base64.b64encode(saveas)	
    return encoded

def bnb_test (variables, datapoints, predictionFeature, rawModel):

    decoded = base64.b64decode(rawModel)
    bnb2 = pickle.loads(decoded)
    predictionList = []
    for i in range (len(datapoints)):
        temp = bnb2.predict(datapoints[i])
        if isinstance (temp,str):
            finalPrediction = {predictionFeature:temp}
        else:
            #here !!
            finalPrediction = {predictionFeature:temp[0]}
        predictionList.append(finalPrediction)
    return predictionList

############END BNB

############BEGIN LASSO
def lasso (variable_values, target_variable_values, alpha = 0):

    clf = linear_model.Lasso(alpha)

    clf.fit (variable_values, target_variable_values) 
    saveas = pickle.dumps(clf)

    encoded = base64.b64encode(saveas)	
    return encoded

def lasso_test (variables, datapoints, predictionFeature, rawModel):

    decoded = base64.b64decode(rawModel)
    clf2 = pickle.loads(decoded)
    predictionList = []
    for i in range (len(datapoints)):
        temp = clf2.predict(datapoints[i])
        finalPrediction = {predictionFeature:temp[0]}
        predictionList.append(finalPrediction)
    return predictionList

############END LASSO

#duplicate
"""
def getJsonContentsTrain (jsonInput):
    try:
        dataset = jsonInput["dataset"]
        predictionFeature = jsonInput["predictionFeature"]
        parameters = jsonInput["parameters"]

        datasetURI = dataset.get("datasetURI", None)
        dataEntry = dataset.get("dataEntry", None)

        variables = dataEntry[0]["values"].keys() 

        datapoints =[]
        target_variable_values = []
        for i in range(len(dataEntry)):
            datapoints.append([])

        for i in range(len(dataEntry)):
            for j in variables:		
                if j == predictionFeature:
                    target_variable_values.append(dataEntry[i]["values"].get(j))
                else:
                    datapoints[i].append(dataEntry[i]["values"].get(j))				

        variables.remove(predictionFeature)		
			
    except(ValueError, KeyError, TypeError):
        print "Error: Please check JSON syntax... \n"
	
    return variables, datapoints, predictionFeature, target_variable_values, parameters

def getJsonContentsTest (jsonInput):
    try:
        dataset = jsonInput["dataset"]	
        rawModel = jsonInput["rawModel"]
        additionalInfo = jsonInput["additionalInfo"]
		
        datasetURI = dataset.get("datasetURI", None)
        dataEntry = dataset.get("dataEntry", None)
		
        predictionFeature = additionalInfo[0].get("predictedFeature", None)
		
        variables = dataEntry[0]["values"].keys() 
	
        datapoints =[]
        for i in range(len(dataEntry)):
            datapoints.append([])
			
        for i in range(len(dataEntry)):
            for j in variables:		
                datapoints[i].append(dataEntry[i]["values"].get(j))				
		
    except(ValueError, KeyError, TypeError):
        print "Error: Please check JSON syntax... \n"
		
    return variables, datapoints, predictionFeature, rawModel

"""
# end duplicate
def entropy(data, attribute):
    value_frequencies = {}
    data_entropy = 0.0

    for record in data:
        if (value_frequencies.has_key(record[attribute])):
            value_frequencies[record[attribute]] += 1.0  
        else:
            value_frequencies[record[attribute]] = 1.0
			
    for frequency in value_frequencies.values():
        data_entropy += (-frequency/len(data)) * math.log(frequency/len(data), 2)   
    
    return data_entropy

def information_gain(data, attribute, target_attribute):

    value_frequencies = {}
    subset_entropy = 0.0

    for record in data:
        if (value_frequencies.has_key(record[attribute])):
            value_frequencies[record[attribute]] += 1.0
        else:
            value_frequencies[record[attribute]] = 1.0
			
    for value in value_frequencies.keys():
        value_prior = value_frequencies[value] / sum(value_frequencies.values())    
        data_subset = [record for record in data if record[attribute] == value]
        subset_entropy += value_prior * entropy(data_subset, target_attribute)
	
    information = entropy(data, target_attribute) - subset_entropy
	
    return information  

#begin duplicate
"""
def unique(lst):
    lst = lst[:]
    unique_lst = []

    for item in lst:
        if unique_lst.count(item) <= 0:
            unique_lst.append(item)
            
    return unique_lst

def get_values(data, attr):
    data = data[:]
    return unique([record[attr] for record in data])

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def majority_value(data, target_attr):
    data = data[:]
    return most_frequent([record[target_attr] for record in data])

def most_frequent(lst):
    lst = lst[:]
    highest_freq = 0
    most_freq = None

    for val in unique(lst):
        if lst.count(val) > highest_freq:
            most_freq = val
            highest_freq = lst.count(val)
            
    return most_freq	

def get_examples(data, attr, value):
    data = data[:]
    rtn_lst = []
    
    if not data:
        return rtn_lst
    else:
        for record in data:
            if record[attr] == value:
                rtn_lst.append(record)    
        return rtn_lst
"""
#end duplicate

def numerical_information_gain(data, attribute, target_attribute):
    value_frequencies = {}
    subset_entropy = 0.0
	
    for record in data:
        if (value_frequencies.has_key(record[attribute])):
            value_frequencies[record[attribute]] += 1.0
        else:
            value_frequencies[record[attribute]] = 1.0
    
    
    all_numeric = get_values(data, attribute)
    all_numeric = [float(num) for num in all_numeric]
    all_numeric.sort()

    best_gain = 0.0
    best_split_number = 0.0

    for numeric_index in range (1,len(all_numeric)-1): 
        temp = deepcopy(data)
        for record in temp:
            if float(record[attribute]) >= all_numeric[numeric_index]:
                record[attribute] = ">="+str(all_numeric[numeric_index])
            else:
                record[attribute] = "<"+str(all_numeric[numeric_index])

        gain = information_gain(temp, attribute, target_attribute)

        if (gain >= best_gain and attribute != target_attribute):
            best_gain = gain
            best_split_number = all_numeric[numeric_index]	

    temp = deepcopy(data)		
    for record in temp:
        if float(record[attribute]) >= best_split_number:
            record[attribute] = ">="+str(best_split_number)
        else:
            record[attribute] = "<"+str(best_split_number)
    return [temp, best_split_number]

def mutual_information(data ,x_index, y_index, logBase, debug = False): 

    summation = 0.0

    values_x = set([data[i].get(x_index) for i in range (len(data))])
    values_lx = list([data[i].get(x_index) for i in range (len(data))])
        
    values_y = set([data[i].get(y_index) for i in range (len(data))])
    values_ly = list([data[i].get(y_index) for i in range (len(data))])
        
    for value_x in values_x:
        for value_y in values_y:
            px = values_lx.count(value_x) / len(data)
            py = values_ly.count(value_y) / len(data)
            
            indexesX = [i for i,x in enumerate(values_lx) if x == value_x]
            indexesY = [i for i,y in enumerate(values_ly) if y == value_y]

            pxy = len(where(in1d(indexesX, indexesY)==True)[0] ) / len(data) 

            if pxy > 0.0:
                summation += pxy * math.log((pxy / (px*py)), logBase)

    return summation

def decision_tree(data, attributes, target_attribute, algorithm, logBase = 10): 

    class_values = [record[target_attribute] for record in data]
    default = majority_value(data, target_attribute)

    if not data or (len(attributes) - 1) <= 0:
        return default
    elif class_values.count(class_values[0]) == len(class_values):
        return class_values[0]
    else:
        if algorithm == "id3":
            best_gain = 0.0
            best_attr = None
			
            for attribute in attributes:
                unique_datapoints = []

                unique_datapoints = [record[attribute] for record in data]
				
				
                temp = []	
                best_split_number = 0.0				

                if len(set(unique_datapoints)) > 2:
                    temp, best_split_number = numerical_information_gain(data, attribute, target_attribute) 
                    data = deepcopy(temp)
                gain = information_gain(data, attribute, target_attribute)
				
                if (gain >= best_gain and attribute != target_attribute):
                    best_gain = gain
                    best_attr = attribute
         
            
            
            unique = [record[best_attr] for record in data]
            unique = list(set(unique)) 
			
            if len(unique)>1 and best_gain >0:
                tree = {best_attr:{}}
                for value in unique: 
                    subtree = decision_tree( get_examples(data, best_attr, value), [attr for attr in attributes if attr != best_attr], target_attribute, algorithm)
                    tree[best_attr][value] = subtree
            else:
                tree = default

        if algorithm == "mci":
            best_gain = 0.0
            best_attr = None
			
            for attribute in attributes:
                unique_datapoints = []

                unique_datapoints = [record[attribute] for record in data]

                temp = []	
                best_split_number = 0.0				
				
                if len(set(unique_datapoints)) > 2:
                    temp, best_split_number = numerical_information_gain(data, attribute, target_attribute) 
                    data = deepcopy(temp)

                gain = mutual_information(data ,attribute, target_attribute, logBase, debug = False)

                if (gain >= best_gain and attribute != target_attribute):
                    best_gain = gain
                    best_attr = attribute           
            
            unique = [record[best_attr] for record in data]
            unique = list(set(unique)) 

            if len(unique)>1 and best_gain >0:
                tree = {best_attr:{}}
                for value in unique: 
                    subtree = decision_tree( get_examples(data, best_attr, value), [attr for attr in attributes if attr != best_attr], target_attribute, algorithm)
                    tree[best_attr][value] = subtree
            else:
                tree = default

    return tree # ++ attr. used

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_in_list(myList, myKey):
    try:
        myList[myKey]
        return True
    except KeyError:
        return False
		
def dicreader(dic, attributes, data_instance):
    if isinstance(dic, dict):
        for k,v in dic.items():
            if k in attributes:
                test_value = data_instance[k]
                if not is_in_list(v,test_value):
                    for value in v:
                        if ">=" in str(value):
                            if float(test_value) >= float(str(value).strip(">=")):
                                test_value = value
                                break
                        if "<" in str(value):
                            if float(test_value) < float(str(value).strip("<")):
                                test_value = value
                                break

                return dicreader(v[test_value], attributes, data_instance)
    else:
        return dic

def tree_predict(variables, datapoints, predictionFeature, rawModel):
    decoded = base64.b64decode(rawModel)
    modeldata = ast.literal_eval(decoded)
    predictionList = []

    for data_instance in datapoints:
        pred = dicreader (modeldata, variables, data_instance)
        finalPrediction = {predictionFeature:pred} 
        predictionList.append(finalPrediction)

    return predictionList
    

def prune (dic, attr):

    if isinstance(dic, dict):
        for k in dic.keys():
            temp = []          
            if (k in attr) and (isinstance(dic[k], dict)):
                for v in dic[k].keys():
                    if isinstance(dic[k][v], dict):
                        return prune(dic[k][v],attr)
                    else:
                        temp.append(dic[k][v])
                        splitVal = v
            else:
                splitVal = k  

            if len(temp)>1 and len(list(set(temp))) ==1:
                return prune([str(dic), str(dic[k][v])],attr)
       
    else:
        return dic

def changeDicVal (dic):
    checker = 0 #       
    while 1:
        try:
            if isinstance(dic, dict):
               checker = 1 #
            else:
                dic = ast.literal_eval(dic)

            params = prune(dic, attr)

            if params is not None:
                if len(params) ==2 and isinstance(params, list):
                    dic = str(dic).replace(params[0], "'"+params[1]+"'")
            else:
                break
        except:
            break        
    return dic

########################################################
########################################################

@app.errorhandler(500)
def not_found(error):
    return make_response(jsonify( { 'error': 'This is it' } ), 500)

@app.errorhandler(400)
def not_found(error):
    return make_response(jsonify( { 'error': 'Bad request' } ), 400)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify( { 'error': 'Not found' } ), 404)

@app.route('/pws/vip/train', methods = ['POST']) ##
def create_task_vip_train():
    start_time = time.time()
    print "IN"
    ###original
    if not request.json:
        abort(400)
    variables, datapoints, predictionFeature, target_variable_values, parameters = getJsonContentsTrain(request.json)

    """
    if not request.environ['body_copy']:
        abort(500)
    
    myTask = request.environ['body_copy']
    readThis = json.dumps(myTask)
    readThis = readThis.replace('\\"','"')
    readThis = readThis.replace('"{','{')
    readThis = readThis.replace('}"','}')
    readThis = json.loads(readThis)
    variables, datapoints, predictionFeature, target_variable_values, parameters = getJsonContentsTrain(readThis)
    """

    latent_variables = parameters.get("latentVariables", None)
	
    Xcopy = deepcopy(datapoints) 
    Ycopy = deepcopy(target_variable_values) 
    Vcopy = deepcopy(variables) 
	
    ## working
    vipMatrix = plsvip(datapoints,target_variable_values,variables,latent_variables)
    encoded, X_new, V_new, lv_best = bestpls(vipMatrix, Xcopy, Ycopy, Vcopy) 

    #DEBUG
    #pls2 = linear_model.LinearRegression()
    #pls2 = PLSRegression(n_components=2,scale=False) #, tol=1e-06
    #pls2.fit(Xcopy, Ycopy)
    #Xcopy1, Ycopy1 = pls2.transform(Xcopy, Ycopy)
    #Ypred = pls2.predict(Xcopy)
    #print pls2.score(Xcopy, Ycopy)

    predictedString = predictionFeature + " predicted"
    task = {
        "rawModel": encoded,
        "pmmlModel": "", 
        "additionalInfo" : [{'predictedFeature': predictedString, 'latentVariables': lv_best}], 
        "independentFeatures": V_new, 
        "predictedFeatures": [predictedString] 
    }
    jsonOutput = jsonify( task )
    #xx = open("vipResponse", "w")
    #xx.writelines(str(vipMatrix)) ####
    #xx.close()
    print"--- %s seconds ---" % (time.time() - start_time)
    return jsonOutput, 201 

@app.route('/pws/vip/test', methods = ['POST'])
def create_task_vip_test():

    if not request.json:
        abort(400)

    variables, datapoints, predictionFeature, rawModel = getJsonContentsTest(request.json)
    
    #print "\n\n LIST OF STUFF", len(datapoints), len(datapoints[0]), len(variables), predictionFeature, "\n\n"
    predictionList = plsvip_predict(variables, datapoints, predictionFeature, rawModel)
    task = {
        "predictions": predictionList
    }
    jsonOutput = jsonify( task )

    #xx = open("vipPred", "w")
    #xx.writelines(str(predictionList))
    #xx.close()

    return jsonOutput, 201 

@app.route('/pws/lm/train', methods = ['POST'])
def create_task_lm_train():
    if not request.json: # original
        abort(400)       # original 

    variables, datapoints, predictionFeature, target_variable_values, parameters = getJsonContentsTrain(request.json) # original

    ## 1
    #if not request.get_json(force=True, silent=True): #debug 28032016
    #    abort(400) #debug 28032016
    
    ## 2
    #data = request.get_data()
    #if not data:
    #    abort(500)
    #print data 
    #variables, datapoints, predictionFeature, target_variable_values, parameters = getJsonContentsTrain(request.get_json(force=True, silent=True)) 

    encoded = lm(datapoints, target_variable_values)
	
    predictedString = predictionFeature + " predicted"
    task = {
        "rawModel": encoded,
        "pmmlModel": "", 
        "additionalInfo" : [{'predictedFeature': predictedString}], 
        "independentFeatures": variables, 
        "predictedFeatures": [predictedString] 
    }
    #task = {} #debug 28032016
    jsonOutput = jsonify( task )
    #xx = open("lmResponse", "w")
    #xx.writelines(str(encoded))
    #xx.close()
    return jsonOutput, 201 

@app.route('/pws/lm/test', methods = ['POST'])
def create_task_lm_test():

    if not request.json:
        abort(400)

    variables, datapoints, predictionFeature, rawModel = getJsonContentsTest(request.json)
    predictionList = lm_test(variables, datapoints, predictionFeature, rawModel)
	
    task = {
        "predictions": predictionList
    }
    jsonOutput = jsonify( task )
    #xx = open("lmTest", "w")
    #xx.writelines(str(predictionList))
    #xx.close()
    return jsonOutput, 201 

@app.route('/pws/lasso/train', methods = ['POST'])
def create_task_lasso_train():

    if not request.json:
        abort(400)

    variables, datapoints, predictionFeature, target_variable_values, parameters = getJsonContentsTrain(request.json)
    alpha = parameters.get("alpha", None)
    if (alpha):
        encoded = lasso(datapoints, target_variable_values, alpha = 0.1)
    else:
        encoded = lasso(datapoints, target_variable_values)
	
    predictedString = predictionFeature + " predicted"
    task = {
        "rawModel": encoded,
        "pmmlModel": "", 
        "additionalInfo" : [{'predictedFeature': predictedString}], 
        "independentFeatures": variables, 
        "predictedFeatures": [predictedString] 
    }
    jsonOutput = jsonify( task )
    #xx = open("lassoResponse", "w")
    #xx.writelines(str(encoded))
    #xx.close()
    return jsonOutput, 201 

@app.route('/pws/lasso/test', methods = ['POST'])
def create_task_lasso_test():

    if not request.json:
        abort(400)

    variables, datapoints, predictionFeature, rawModel = getJsonContentsTest(request.json)
    predictionList = lasso_test(variables, datapoints, predictionFeature, rawModel)
	
    task = {
        "predictions": predictionList
    }
    jsonOutput = jsonify( task )
    #xx = open("lassoTest", "w")
    #xx.writelines(str(predictionList))
    #xx.close()
    return jsonOutput, 201 


@app.route('/pws/id3/train', methods = ['POST'])
def create_task_id3_train():

    if not request.json:
        abort(400)

    variables, datapoints, predictionFeature, target_variable_values, parameters = getJsonContentsTrain(request.json)
    logBase = parameters.get("logBase", None)
	
 
    if is_number(target_variable_values[0]):
        bin_width = 3.5*(numpy.std(target_variable_values))*(numpy.power(len(target_variable_values), -0.3))
        num_bins = math.ceil(max(target_variable_values) - min(target_variable_values)/bin_width)
        num_bins = int(round(num_bins))
		
        bin_values = []
        bin_diff = (max(target_variable_values) - min(target_variable_values)) / num_bins
        
        temp1 = min(target_variable_values)
        temp2 = min(target_variable_values) + bin_diff

        for i in range (0, num_bins):
            bin_values.append ([temp1, temp2])
            temp1 += bin_diff
            temp2 += bin_diff
        bin_values[len(bin_values) - 1][1] += 0.000001 

        for i in range (len (target_variable_values)):
            for j in range (len (bin_values)):
                if target_variable_values[i] >= bin_values[j][0] and target_variable_values[i] < bin_values[j][1]:
                    target_variable_values[i] = "["+str(bin_values[j][0])+","+str(bin_values[j][1])+")"
                    break

			
    data = []
    for instance in range (len(datapoints)):
        attributePerRowDictionary = {}

        for variable in range (len(variables)): 
            attributePerRowDictionary[variables[variable]] = datapoints[instance][variable]
			
        attributePerRowDictionary[predictionFeature] = target_variable_values[instance]
		
        data.append(attributePerRowDictionary)


    prediction = decision_tree(data, variables, predictionFeature, "id3")  

    dic = ast.literal_eval(str(prediction))
    newDic = changeDicVal(dic)

    encoded = base64.b64encode(str(newDic))

    predictedString = predictionFeature + " predicted"
    task = {
        "rawModel": encoded,
        "pmmlModel": "", 
        "additionalInfo" : [{'predictedFeature': predictedString}], 
        "independentFeatures": variables, 
        "predictedFeatures": [predictedString] 
    }
    jsonOutput = jsonify( task )
    #xx = open("id3Response", "w")
    #xx.writelines(str(encoded))
    #xx.close()
    return jsonOutput, 201 

@app.route('/pws/id3/test', methods = ['POST'])
def create_task_id3_test():

    if not request.json:
        abort(400)

    variables, datapoints, predictionFeature, rawModel = getJsonContentsTest(request.json)

    data = []
    for instance in range (len(datapoints)):
        attributePerRowDictionary = {}

        for variable in range (len(variables)): 
            attributePerRowDictionary[variables[variable]] = datapoints[instance][variable]
		
        data.append(attributePerRowDictionary)
		
    predictionList = tree_predict(variables, data, predictionFeature, rawModel) 
	
    task = {
        "predictions": predictionList
    }
    jsonOutput = jsonify( task )
    #xx = open("id3Test", "w")
    #xx.writelines(str(predictionList))
    #xx.close()
    return jsonOutput, 201 

@app.route('/pws/mci/train', methods = ['POST']) 
def create_task_mci_train():

    if not request.json:
        abort(400)

    variables, datapoints, predictionFeature, target_variable_values, parameters = getJsonContentsTrain(request.json)
    logBase = parameters.get("logBase", None)
    if not (logBase):
        logBase = 10
    if is_number(target_variable_values[0]):
        bin_width = 3.5*(numpy.std(target_variable_values))*(numpy.power(len(target_variable_values), -0.3)) #Scott
        num_bins = math.ceil(max(target_variable_values) - min(target_variable_values)/bin_width)
        num_bins = int(round(num_bins))
		
        bin_values = []
        bin_diff = (max(target_variable_values) - min(target_variable_values)) / num_bins
        
        temp1 = min(target_variable_values)
        temp2 = min(target_variable_values) + bin_diff

        for i in range (0, num_bins):
            bin_values.append ([temp1, temp2])
            temp1 += bin_diff
            temp2 += bin_diff
        bin_values[len(bin_values) - 1][1] += 0.000001 

        for i in range (len (target_variable_values)):
            for j in range (len (bin_values)):
                if target_variable_values[i] >= bin_values[j][0] and target_variable_values[i] < bin_values[j][1]:
                    target_variable_values[i] = "["+str(bin_values[j][0])+","+str(bin_values[j][1])+")"
                    break

    data = []
    for instance in range (len(datapoints)):
        attributePerRowDictionary = {}

        for variable in range (len(variables)): 
            attributePerRowDictionary[variables[variable]] = datapoints[instance][variable]
			
        attributePerRowDictionary[predictionFeature] = target_variable_values[instance]
		
        data.append(attributePerRowDictionary)

    prediction = decision_tree(data, variables, predictionFeature, "mci", logBase)

    dic = ast.literal_eval(str(prediction))
    newDic = changeDicVal(dic)

    encoded = base64.b64encode(str(newDic))

    predictedString = predictionFeature + " predicted"
    task = {
        "rawModel": encoded,
        "pmmlModel": "", 
        "additionalInfo" : [{'predictedFeature': predictedString, 'logBase': logBase}], 
        "independentFeatures": variables, 
        "predictedFeatures": [predictedString] 
    }
    jsonOutput = jsonify( task )
    #xx = open("mciResponse", "w")
    #xx.writelines(str(encoded))
    #xx.close()
    return jsonOutput, 201 

@app.route('/pws/mci/test', methods = ['POST'])
def create_task_mci_test():

    if not request.json:
        abort(400)

    variables, datapoints, predictionFeature, rawModel = getJsonContentsTest(request.json)

    data = []
    for instance in range (len(datapoints)):
        attributePerRowDictionary = {}

        for variable in range (len(variables)): 
            attributePerRowDictionary[variables[variable]] = datapoints[instance][variable]
		
        data.append(attributePerRowDictionary)
		
    predictionList = tree_predict(variables, data, predictionFeature, rawModel) 
	
    task = {
        "predictions": predictionList
    }
    jsonOutput = jsonify( task )
    #xx = open("mciTest", "w")
    #xx.writelines(str(predictionList))
    #xx.close()
    return jsonOutput, 201 
	
@app.route('/pws/gnb/train', methods = ['POST'])
def create_task_gnb_train():

    if not request.json:
        abort(400)

    variables, datapoints, predictionFeature, target_variable_values, parameters = getJsonContentsTrain(request.json)
    encoded = gnb(datapoints, target_variable_values)
	
    predictedString = predictionFeature + " predicted"
    task = {
        "rawModel": encoded,
        "pmmlModel": "", 
        "additionalInfo" : [{'predictedFeature': predictedString}], 
        "independentFeatures": variables, 
        "predictedFeatures": [predictedString] 
    }
    jsonOutput = jsonify( task )
    #xx = open("gnbResponse", "w")
    #xx.writelines(str(encoded))
    #xx.close()
    return jsonOutput, 201 

@app.route('/pws/gnb/test', methods = ['POST'])
def create_task_gnb_test():

    if not request.json:
        abort(400)

    variables, datapoints, predictionFeature, rawModel = getJsonContentsTest(request.json)
    predictionList = gnb_test(variables, datapoints, predictionFeature, rawModel)
	
    task = {
        "predictions": predictionList
    }
    jsonOutput = jsonify( task )
    #xx = open("gnbTest", "w")
    #xx.writelines(str(predictionList))
    #xx.close()
    return jsonOutput, 201 

@app.route('/pws/mnb/train', methods = ['POST'])
def create_task_mnb_train():

    if not request.json:
        abort(400)

    variables, datapoints, predictionFeature, target_variable_values, parameters = getJsonContentsTrain(request.json)
    encoded = mnb(datapoints, target_variable_values)
	
    predictedString = predictionFeature + " predicted"
    task = {
        "rawModel": encoded,
        "pmmlModel": "", 
        "additionalInfo" : [{'predictedFeature': predictedString}], 
        "independentFeatures": variables, 
        "predictedFeatures": [predictedString] 
    }
    jsonOutput = jsonify( task )
    #xx = open("gnbResponse", "w")
    #xx.writelines(str(encoded))
    #xx.close()
    return jsonOutput, 201 

@app.route('/pws/mnb/test', methods = ['POST'])
def create_task_mnb_test():

    if not request.json:
        abort(400)

    variables, datapoints, predictionFeature, rawModel = getJsonContentsTest(request.json)
    predictionList = mnb_test(variables, datapoints, predictionFeature, rawModel)
	
    task = {
        "predictions": predictionList
    }
    jsonOutput = jsonify( task )
    #xx = open("gnbTest", "w")
    #xx.writelines(str(predictionList))
    #xx.close()
    return jsonOutput, 201 

@app.route('/pws/bnb/train', methods = ['POST'])
def create_task_bnb_train():

    if not request.json:
        abort(400)

    variables, datapoints, predictionFeature, target_variable_values, parameters = getJsonContentsTrain(request.json)
    encoded = bnb(datapoints, target_variable_values)
	
    predictedString = predictionFeature + " predicted"
    task = {
        "rawModel": encoded,
        "pmmlModel": "", 
        "additionalInfo" : [{'predictedFeature': predictedString}], 
        "independentFeatures": variables, 
        "predictedFeatures": [predictedString] 
    }
    jsonOutput = jsonify( task )
    #xx = open("gnbResponse", "w")
    #xx.writelines(str(encoded))
    #xx.close()
    return jsonOutput, 201 

@app.route('/pws/bnb/test', methods = ['POST'])
def create_task_bnb_test():

    if not request.json:
        abort(400)

    variables, datapoints, predictionFeature, rawModel = getJsonContentsTest(request.json)
    predictionList = bnb_test(variables, datapoints, predictionFeature, rawModel)
	
    task = {
        "predictions": predictionList
    }
    jsonOutput = jsonify( task )
    #xx = open("gnbTest", "w")
    #xx.writelines(str(predictionList))
    #xx.close()
    return jsonOutput, 201 

############################################################
#plan B

from werkzeug.wsgi import LimitedStream

class StreamConsumingMiddleware(object):

    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        print "HERE"
        stream = LimitedStream(environ['wsgi.input'],  0) # int(environ['CONTENT_LENGTH'] or 0))

        print stream
        environ['wsgi.input'] = stream
        app_iter = self.app(environ, start_response)
        try:
            stream.exhaust()
            for event in app_iter:
                yield event
        finally:
            if hasattr(app_iter, 'close'):
                app_iter.close()
        return app_iter
############################################################
# plan A
"""
class WSGICopyBody(object):
    def __init__(self, application):
        self.application = application

    def __call__(self, environ, start_response):
        from cStringIO import StringIO
        input = environ.get('wsgi.input')
        length = environ.get('CONTENT_LENGTH', '0')
        length = 0 if length == '' else int(length)
        body = ''
        if length == 0:
            environ['body_copy'] = ''
            if input is None:
                return
            if environ.get('HTTP_TRANSFER_ENCODING','0') == 'chunked':
                size = int(input.readline(),16)
                while size > 0:
                    temp = str(input.read(size+2)).strip()
                    body += temp
                    size = int(input.readline(),16)
        else:
            body = environ['wsgi.input'].read(length)
        environ['body_copy'] = body
        environ['wsgi.input'] = StringIO(body)

        # Call the wrapped application
        app_iter = self.application(environ, 
                                    self._sr_callback(start_response))

        # Return modified response
        print app_iter
        return app_iter

    def _sr_callback(self, start_response):
        def callback(status, headers, exc_info=None):

            # Call upstream start_response
            start_response(status, headers, exc_info)
        print callback
        return callback
"""
############################################################

if __name__ == '__main__': 
    #app.wsgi_app = WSGICopyBody(app.wsgi_app) # plan A
    app.wsgi_app = StreamConsumingMiddleware(app.wsgi_app) # plan B
    app.run(host="0.0.0.0", port = 5000, debug = True)	

#curl -i -H "Content-Type: application/json" -X POST -d @C:/Python27/Flask-0.10.1/python-api/vipbugtrain.json http://localhost:5000/pws/vip/train