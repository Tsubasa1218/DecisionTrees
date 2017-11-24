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

    tree = Node()
    tree.value = data_set_attributes

    Preprocessing(tree)


def Preprocessing(tree):
    #This part tries to define the set of labels that are going to be evaluated 
    #Only for this case, where we are evaluating only the second column

    #TODO: Scale this code to be used in N columns. DONE!!!
    #What it finally does is agrouping tha characteristics in lists
    labels_list = []
    for column in range(0, len(tree.value[0])):
        for i in tree.value:
            if(i[column] not in labels_list):
                labels_list.append(i[column])
    print(labels_list)
    
    label_groups = [[]] * len(tree.value[0])
    for column in range(0, len(tree.value[0])):
        aux_list = []
        for label in labels_list:
            for key in tree.value:        
                if(label == key[column]):
                    aux_list.append(label)
                    break
        label_groups[column] = aux_list           

    print(label_groups)

    #This section is used to count how many characteristics are in the list
    #To proceed to calculate Entropy values for each characteristic
    label_frequency_list = [0]*len(labels_list)
    
    for element in tree.value:
        for i in range(0, len(labels_list)):
            for j in range(0, len(element)):
                if(element[j] == labels_list[i]):
                    label_frequency_list[i] += 1
    print(label_frequency_list)
    
    #Now that we have the frequencies, can apply the formula to calculate entropy
    #entropy_value = Entropy(label_frequency_list)
    #print(entropy_value)
        
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

