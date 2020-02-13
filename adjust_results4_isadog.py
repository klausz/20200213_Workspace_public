#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/adjust_results4_isadog.py
#                                                                             
# PROGRAMMER: 
# DATE CREATED:                                 
# REVISED DATE: 
# PURPOSE: Create a function adjust_results4_isadog that adjusts the results 
#          dictionary to indicate whether or not the pet image label is of-a-dog, 
#          and to indicate whether or not the classifier image label is of-a-dog.
#          All dog labels from both the pet images and the classifier function
#          will be found in the dognames.txt file. We recommend reading all the
#          dog names in dognames.txt into a dictionary where the 'key' is the 
#          dog name (from dognames.txt) and the 'value' is one. If a label is 
#          found to exist within this dictionary of dog names then the label 
#          is of-a-dog, otherwise the label isn't of a dog. Alternatively one 
#          could also read all the dog names into a list and then if the label
#          is found to exist within this list - the label is of-a-dog, otherwise
#          the label isn't of a dog. 
#         This function inputs:
#            -The results dictionary as results_dic within adjust_results4_isadog 
#             function and results for the function call within main.
#            -The text file with dog names as dogfile within adjust_results4_isadog
#             function and in_arg.dogfile for the function call within main. 
#           This function uses the extend function to add items to the list 
#           that's the 'value' of the results dictionary. You will be adding the
#           whether or not the pet image label is of-a-dog as the item at index
#           3 of the list and whether or not the classifier label is of-a-dog as
#           the item at index 4 of the list. Note we recommend setting the values
#           at indices 3 & 4 to 1 when the label is of-a-dog and to 0 when the 
#           label isn't a dog.
#
##
# TODO 4: Define adjust_results4_isadog function below, specifically replace the None
#       below by the function definition of the adjust_results4_isadog function. 
#       Notice that this function doesn't return anything because the 
#       results_dic dictionary that is passed into the function is a mutable 
#       data type so no return is needed.
# 
def adjust_results4_isadog(results_dic, dogfile):
    """
    Adjusts the results dictionary to determine if classifier correctly 
    classified images 'as a dog' or 'not a dog' especially when not a match. 
    Demonstrates if model architecture correctly classifies dog images even if
    it gets dog breed wrong (not a match).
    Parameters:
      results_dic - Dictionary with 'key' as image filename and 'value' as a 
                    List. Where the list will contain the following items: 
                  index 0 = pet image label (string)
                  index 1 = classifier label (string)
                  index 2 = 1/0 (int)  where 1 = match between pet image
                    and classifer labels and 0 = no match between labels
                ------ where index 3 & index 4 are added by this function -----
                 NEW - index 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                 NEW - index 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
     dogfile - A text file that contains names of all dogs from the classifier
               function and dog names from the pet image files. This file has 
               one dog name per line dog names are all in lowercase with 
               spaces separating the distinct words of the dog name. Dog names
               from the classifier function can be a string of dog names separated
               by commas when a particular breed of dog has multiple dog names 
               associated with that breed (ex. maltese dog, maltese terrier, 
               maltese) (string - indicates text file's filename)
    Returns:
           None - results_dic is mutable data type so no return needed.
    """    
    #print('results_dic_1st_______________________', results_dic)
#    None
    #d = {}
    #with open("dict.txt") as f:
    #for line in f:
    
    #filename = 'commands.txt'
    #commands = {}
    #with open(filename) as fh:
        #for line in fh:
            #command, description = line.strip().split(' ', 1)
            #commands[command] = description.strip()

#dognames_dic = { line.split(" ")[0] : line[0:] for line in open('dognames.txt') }
#def readFile(filename):
#    # Dict that will contain keys and values

    filename = 'dognames.txt'
    counter = int(0)
#dictionary =  {}
    docnames_dic = {}
    with open(filename, "r") as f:
        for line in f:
            dog_name_arr = line.replace('\n','').split(',')
            for dog_name in dog_name_arr:
                if dog_name.lstrip() not in docnames_dic.values():
                    docnames_dic[dog_name.lstrip()] = counter #s[0:] #the dogname is in the key field
                    #print('dog_name added:', dog_name.lstrip())
                #else: 
                    #print('dog_name was already in dictionary - no action')
            counter += 1
#        return dictionary
    for key, values in results_dic.items():
#    for keys, values in results_dic.items():
        #        print('values[0] (results_dic - by record):' , values[0]) 
        #print('key+values (results_dic - by record):' , keys, values[0])
        #if results_dic[key][1] in dognames_dic[1]:
        #print('results_dic__key: ', key, '    value[0]: ',values[0])

        
#Whats used in file classify images in dictionary of lists:
#        dictionary_of_lists[filename] = [results_dic[filename]
#                                            , image_classification.lower()
#                                            , int(results_dic[filename] in image_classification.lower())]
        for key in docnames_dic:
            #print('docnames_dic__key: ', key)
            if key == values[0]:
                #print('gefunden', key, ' == ' , values[0])
                #results_dic[key].extend((1, 1))
                break
            #else:
                #print('suche noch')
#        print('result-DogFilename: ',values[0], '-- KeyFromDogfile: ',docnames_dic['german shepherd dog']) #,docnames_dic[values[0]])
#                st = "ABSTTHGIHG"
#print dict[st[-9:]]
#    for keys, values in docnames_dic.items():
#        print('key (docnames_dic: -by record):' , keys) 
        #print('key+values (docnames_dic: -by record):' , keys, values) 
    #print()
    #print('dognames_dic_______________________', docnames_dic)
    #print()
    #print('results_dic________________________', results_dic) #results_dic seems to have the same content as dictionary of lists, so, if this is enhanced with the additional 2 binary fields?

    #print()
    #print('docnames_dic(adjustResults4_isadog): ', docnames_dic)
    #print()
    #print('results_dic__AGAINST_FILE:', str(results_dic))
    #print()
    #print('dog_in_dog:', 'german shepherd dog' in 'german shepherd dog')
    #print ('dog_in_dog__LONG_STRING:', ('german shepherd dog', 'german shepherd, german shepherd dog, german police dog, alsatian' in 'german shepherd dog'))
    #print('docnames_dic__FILE:', docnames_dic)
    for key,values in results_dic.items():
        if (results_dic[key][0] not in docnames_dic): # not declared as dog, its not a dog - no tricks
            results_dic[key].extend((0,0))
            #print(results_dic[key][0], '0x gefunden -> extend 0,0')
            #break
        else:
            if (results_dic[key][0] in docnames_dic): 
                #results_dic[key].extend((1,0)) #it will always be set and maybe overwritten with 1,1 
                #print(results_dic[key][0] , 'min 1x gefunden')
                tmp_list = list()
                tmp_list = results_dic[key][1].split(',') #as I have multiple entries 
                #print('---AUSWAHL_tmp_list: ',tmp_list)
                for x in tmp_list:
                    #print('--AUSWAHL_each_list_element:',x)
                    if ((results_dic[key][0] in docnames_dic) and (x in docnames_dic)): #the declared name nd the name fond for the image is in the dog file 1,1 
                        #print(results_dic[key][0] ,'--------2x--gefunden -> extend 1,1 ', x) 
                        results_dic[key].extend((1,1))
                        #print('1x aber ncht 2x gefunden - extend 1,0')
                        break 
            else:
                #print('Achtung Problem 1,0 nicht hinzugefügt')
                results_dic[key].extend((1,0)) #this should not be passed when found 2 times as there is a break

        ##print(key, values[0])
        ##print(key, values[0])
        #if ((results_dic[key][0] in docnames_dic) and (results_dic[key][1] in docnames_dic)): #the declared name nd the name fond for the image is in the dog file 1,1 
        #    #print('2x gefunden')
        #    print('--in2x--results_dic[key][1]:', results_dic[key][1], results_dic[key][1] in docnames_dic)
        #    #print('--in2xReverse--results_dic[key][1]:', results_dic[key][1], docnames_dic in results_dic[key][1])
        #    results_dic[key].extend((1,1))
        #if ((results_dic[key][0] in docnames_dic) and (results_dic[key][1] not in docnames_dic)): #its declared as a don, but it is not a dog - tried to trick
        #    #print('1x gefunden')
        #    print('--in1x--results_dic[key][1]:', results_dic[key][1], results_dic[key][1] in docnames_dic)
        #    results_dic[key].extend((1,0))
        #if ((results_dic[key][0] not in docnames_dic) and (results_dic[key][1] not in docnames_dic)): # not declared as dog, its not a dog - no tricks
        #    #print('0x gefunden')
        #    print('--in0x--results_dic[key][1]:', results_dic[key][1], results_dic[key][1] in docnames_dic)
        #    results_dic[key].extend((0,0))
        #    #print(results_dic)

    return results_dic