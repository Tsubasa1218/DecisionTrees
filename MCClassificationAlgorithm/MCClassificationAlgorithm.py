#This project is made to compare an implementation of a classifier algorithm for Machine Learning

import math
import numpy as np

class Node:
    def __init__(self):
        #Parent
        self.parent = None
        
        #Comparison tag
        self.value = None
        
        #Leaves
        self.left = None
        self.right = None

def Classification():
    data_set_attributes = [[140, 'Smooth'], [130, 'Smooth'], [150, 'Bumpy'], [170, 'Bumpy'], [177, 'Bumpy'], [135, 'Smooth'], [140, 'Smooth']]
    data_set_labels = ['Apple', 'Apple', 'Orange', 'Orange', 'Orange', 'Apple']

    #We have to discretize the values that are continous
    print(data_set_attributes)
    cleaned_data = Discretizer(data_set_attributes)

    tree = Node()
    tree.value = cleaned_data

    label_groups, labels_list = Preprocessing(tree)

    Frecuency(tree, labels_list)

    #info_gainance_of_system = 

def Discretizer(data):
    type_names = []
    
    first_row = data[0]
    i = 0
    for element in first_row:
        if(isinstance(element, int)):
            if('Integer' not in type_names):
                type_names.append(['Integer', i])
        elif(isinstance(element, str)):
            if('String' not in type_names):
                type_names.append(['String', i])
        elif(isinstance(element, float)):
            if('Float' not in type_names):
                type_names.append(['Float', i])
        else:
            if('No type' not in type_names):
                type_names.append(['No type', i])
        i += 1

    #Basic discretization (lol) of integers and floats. We are going to discretize in only 2 sets, based on the median value
    for attr_type in type_names:
        if attr_type[0] == 'Integer' or attr_type[0] == 'Float':
            acum = 0.0
            for row in data:
                try:
                    acum += row[attr_type[1]]
                except TypeError as e:
                    raise(e)
            median = np.floor(acum / len(data))
            #Now we set the new values to the data
            for row in data:
                if(row[attr_type[1]] <= median):
                    row[attr_type[1]] = 'less than ' + str(median)
                else:
                    row[attr_type[1]] = 'greater than ' + str(median)
        
    return data


def Preprocessing(tree):
    #This part tries to define the set of labels that are going to be evaluated 
    #Only for this case, where we are evaluating only the second column <- don't pay attention

    #TODO: Scale this code to be used in N columns. DONE!!!
    #What it finally does is agrouping tha characteristics in lists
    labels_list = []
    for column in range(0, len(tree.value[0])):
        for i in tree.value:
            if(i[column] not in labels_list):
                labels_list.append(i[column])
    #print(labels_list)
    
    label_groups = [[]] * len(tree.value[0])
    for column in range(0, len(tree.value[0])):
        aux_list = []
        for label in labels_list:
            for key in tree.value:        
                if(label == key[column]):
                    aux_list.append(label)
                    break
        label_groups[column] = aux_list           

    #print(label_groups)

    return label_groups, labels_list

def Frecuency(tree, labels_list):
    #This section is used to count how many characteristics are in the list
    #To proceed to calculate Entropy values for each characteristic
    label_frequency_list = [0]*len(labels_list)
    
    for element in tree.value:
        for i in range(0, len(labels_list)):
            for j in range(0, len(element)):
                if(element[j] == labels_list[i]):
                    label_frequency_list[i] += 1
    print(label_frequency_list)
    
        
def Entropy(list):
    #This function calculates the entropy value using:
    #E(x)=-SUM( P(x)*Log2(P(x)) ) ; P is the probability function
    
    entropy_value = 0
    total_frequency = 0
    
    for i in list:
        total_frequency += i
    
    for i in list:
        entropy_value += ProbabilityFunction(i, total_frequency) * math.log2(ProbabilityFunction(i, total_frequency))
    return entropy_value * -1
            

def ProbabilityFunction(ocur, total):
    return ocur/total

def main():
    Classification()

if __name__ == "__main__":
    main() 

