#This project is made to compare an implementation of a classifier algorithm for Machine Learning
import numpy as np

data_set_attributes = [
        [140, 'Smooth', 'Red'],
        [130, 'Smooth', 'Red'], 
        [150, 'Bumpy', 'Yellow'], 
        [170, 'Bumpy', 'Yellow'], 
        [177, 'Bumpy', 'Yellow'], 
        [135, 'Smooth', 'Red'], 
        [140, 'Smooth', 'Red'],
        [144, 'Bumpy', 'Yellow'],
        ]
data_set_labels = ['Apple', 'Apple', 'Orange', 'Orange', 'Orange', 'Apple', 'Apple', 'Orange']
np.random.seed(21)
for i in range(0, 10):
    size = np.random.randint(130, 180)
    rndText = np.random.randint(0, 2)
    if rndText == 0:
        text = 'Smooth'
    else:
        text = 'Bumpy'
    rndColor = np.random.randint(0, 2)
    if rndColor == 0:
        color = 'Red'
        obs = 'Apple'
    else:
        color = 'Yellow'
        obs = 'Orange'
    data_set_attributes.append([size, text, color])
    data_set_labels.append(obs)


class Node:
    def __init__(self):
        #Parent
        self.parent = None
        
        #Comparison tag
        self.value = None
        
        #Leaves
        self.childs_values = []
        self.childs = []

def Classification():
    
    #print(data_set_attributes)
    #print(data_set_labels)

    #We have to discretize the values that are continous
    cleaned_data = Discretizer(data_set_attributes)

    #We get the labels from the data
    label_list = LabelsList(data_set_labels)
    #We calculate the frecuencis
    labels_frequency = Frecuency(data_set_labels, label_list)
    #We calculate the entropy of the system. That is the entropy of the labels
    entropy_of_the_system = Entropy(labels_frequency)
    #print(entropy_of_the_system)
    
    #Now we can start setting up the actual process of finding the nodes
    #Define the root node
    tree = Node()
    tree.value = cleaned_data

    attr_groups, attr_list = Preprocessing(tree)
    
    #Calculate the info. g. of all colums
    entropies = ColumnEntropies(tree.value, data_set_labels, label_list, attr_groups)

    #Find the infoG = EntropyOfSystem - ColumEntropies
    info_gainance = [0]*len(entropies)
    for i in range(0, len(info_gainance)):
        info_gainance[i] = entropy_of_the_system - entropies[i]

    max_info_g = max(info_gainance)
    max_info_pos = info_gainance.index(max_info_g)

    #We have to remove the column from the cleaned_data array/list
    cleaned_data_cp = cleaned_data.copy()
    cleaned_data.clear()
    
    for i in range(0, len(cleaned_data_cp)):
        aux_list = []
        for j in range(0, len(cleaned_data_cp[i])):
            if j != max_info_pos:
                aux_list.append(cleaned_data_cp[i][j])
        cleaned_data.append(aux_list)
    

    #Create of the new nodes
    #for i in range(0, attr_groups[max_info_pos]):
    #    child = Node()
    #    child.parent = tree  
    #    child.value = cleaned_data

    #    tree.childs.append(child)
    #    tree.childs_values.append(attr_groups[max_info_pos][i])
        
    

    



def ColumnEntropies(value_data, label_data, label_list, attr_groups):
     
    info_gainance = [0]*len(attr_groups)
    #This fors calculate the frecuency of an item given the label forr every attribute
    #I.e. how many less than avg are apples and oranges
    #It also calculates the entropy of each column (attribute). Then it calculates the info gainance
    for i in range(0, len(attr_groups)):
        for j in range(0, len(attr_groups[i])):
            #print("se evalua ", attr_groups[i][j])
            frequency = [0]*len(attr_groups[i])
            for k in range(0, len(value_data)):
                if value_data[k][i] == attr_groups[i][j]:
                    for h in range(0, len(label_list)):
                        if label_list[h] == label_data[k]:
                            frequency[h] += 1
            
            #Here we calculate the frequency for each attr value
            column = GetColumnOfMatrix(value_data, i)
            attr_frequency = Frecuency(column, attr_groups[i])
            ocur = attr_frequency[j]
            total = sum(attr_frequency)

            #Here we calculate the info gainance of each attr. The formula is: (ocur/total)*Entropy(frequencies)
            info_gainance[i] += ProbabilityFunction(ocur, total) * Entropy(frequency)
    
    return info_gainance

def GetColumnOfMatrix(value_data, column_pos):
    list = [0]*len(value_data)
    for value in value_data:
        list.append(value[column_pos])
    return list

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

def LabelsList(data):
    labels = []
    for label in data:
        if(label not in labels):
            labels.append(label)

    return labels

def Preprocessing(tree):
    #This part tries to define the set of attributes that are going to be evaluated 
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

def Frecuency(values_list, labels_list):
    #This section is used to count how many characteristics are in the list
    #To proceed to calculate Entropy values for each characteristic
    label_frequency_list = [0]*len(labels_list)
    
    for i in range(0, len(values_list)):
        for j in range(0, len(labels_list)):
            if labels_list[j] == values_list[i]:
                label_frequency_list[j] += 1
    return label_frequency_list
            
def Entropy(list):
    #This function calculates the entropy value using:
    #E(x)=-SUM( P(x)*Log2(P(x)) ) ; P is the probability function
    
    entropy_value = 0
    total_frequency = 0
    
    for value in list:
        total_frequency += value
    
    for i in list:
        a = ProbabilityFunction(i, total_frequency)
        if a > 0:
            b =np.log2(a)
        else:
            b = 0.0
        entropy_value +=  a*b 
    return entropy_value * -1
            

def ProbabilityFunction(ocur, total):
    return ocur/total

def main():
    Classification()

if __name__ == "__main__":
    main() 

