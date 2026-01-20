# -*- coding: utf-8 -*-

# Cole-Kripke algorithm class - 20/09/2019
# Julius Andretti

# References:
# [1] COLE, R. J.; KRIPKE, D.F.; GRUEN, W.; MULLANEY, D. J.; GILLIN, J. C.: Automatic Sleep/Wake Identification from Wrist Activity (1992)
# Default values from [1]: P=0.00001,weights_before=[1408,441,326,598,404],weights_after=[508,350]

import matplotlib.pyplot as plt
from numpy import dot,transpose,asanyarray,where,zeros

class ColeKripke:
    def __init__(self,
                 activity,
                 P=5e-5,
                 weights_before=[230],
                 weights_after=[74, 67, 55],
                 types=[0, 0, 0, 1, 1],
                 max_values=[1, 3, 4, 6, 10],
                 thresh_values=[4, 10, 15, 10, 20],
                 new_state=1,
                 ):
        # activity is the input array
        # P is a scaling factor used on the weigthed sum
        # weights_before[i] are weigthing factors for previous epochs (weights_before[0] is for current epoch, thus len(weights_before) >= 1)
        # weights_after[i] are weigthing factors for future epochs
        # types is an array defining if the rules passed are type 0 or 1
        # max_values and thresh_values contain the definitions of each rule
        # new_state is the state that will be assigned to positions on output array that aren't supposed to keep the previous state

        if (len(weights_before) < 1):
            raise Exception('Array weights_before must have at least one element!')
        
        if ((len(types) != len(max_values)) or (len(types) != len(thresh_values)) or (len(max_values) != len(thresh_values))):
            raise Exception('Arrays types, max_values and thresh_values must have the same size!')
        else:
            for i in range(len(types)):
                if ((types[i] < 0) or (thresh_values[i] < 0) or (max_values[i] < 0)):
                    raise Exception('Elements in arrays types, max_values and thresh_values must be positive integers!')
                elif (types[i] > 1):
                    raise Exception('Elements in array types must be 0 or 1!')        

        self.activity = asanyarray(activity, dtype='float64')   # Activity per minute vector obtained from actigraph data
        
        # weighteded sum configuration 
        self.P = P   # Scale factor
        self.weights_before = asanyarray(weights_before, dtype='float64')   # weights_before[i] are weighting factors for activity at previous minutes
        self.weights_after = asanyarray(weights_after, dtype='float64')   # weights_after[i] are weighting factors for activity at future minutes
        
        # Rescoring configuration 
        self.types = types   # Defines the types (0 or 1) of the  given rules
        self.max_values = max_values   # The maximum number of values that a rule is capable of altering
        self.thresh_values = thresh_values   # After how many repeated values a rule acts
        
        self.new_state = new_state 
        
    def model(self, previous):
        def rescoring_rules(current, change, types, max_values, thresh_values):
            # Type 0 rules are capable of changing up to max_value positions that come after a sequence of length
            # greater than or equal to thresh_value
            def rule_type0(max_value, i):
                for k in range(i, i + max_value):
                    if (k < n):
                        if (current[k] == change):
                            current[k] = ~change
                        else:
                            break
                    else:
                        break
                increment = k - i
                return increment
                
            # Type 1 rules are capable of changing up to max_value positions that come sorrounded by sequences of 
            # length greater than or equal to thresh_value   
            def rule_type1(thresh_value, max_value, i):
                completed = False
                change_count = 1 
                increment = 0
                for j in range(i + 1, n):
                    if (j < (n - 1)):
                        # First, we must determine the length of the sequence of change values
                        if (current[j] == change) and change_count < max_value:
                            change_count += 1
                        else:
                            fix_count = 1
                            if (change_count == max_value):   # If the length of the sequence reaches max_value it must end, so:
                                if (current[j] != change):
                                    # Here, we have a sequence of length max_value and we start to check if there's a sequence of
                                    # ~change values of length at least thresh_value afterwards
                                    for k in range(j + 1, n):
                                        if (k <= (n - 1)):
                                            if ((current[k] != change) and (fix_count < thresh_value)):
                                                fix_count += 1
                                                if (fix_count == thresh_value):   # If there is, the rule is applied
                                                   current[i:(i + change_count)] = ~change
                                                   increment = k - i
                                                   completed = True
                                                   break   # And the k loop is stopped

                                    break   # Algorithm finished
                            else:
                                # This is the case where the sequence of change values has length less than max_value
                                for k in range(j + 1, n):
                                    if (k <= (n - 1)):
                                        if ((current[k] != change) and (fix_count < thresh_value)):
                                            fix_count += 1
                                            if (fix_count == thresh_value): 
                                               current[i:(i+change_count)] = ~change
                                               increment = k - i
                                               completed = True
                                               break

                                break   # Algorithm finished
                return completed, increment
                
            n = len(current)          
            
            # First, we'll separate by rules by type
            type0 = []
            type1 = []
            for i in range(len(types)):
                if types[i]:
                    type1.append([thresh_values[i], max_values[i], 0])   # This last position counts how many times the rule has been applied
                else:
                    type0.append([thresh_values[i], max_values[i], 0])  
                    
            # Then, rules we'll be sorted in ascending order of threshold value
            def get_key(item):
                return item[0]
            type0 = sorted(type0, key=get_key)
            type1 = sorted(type1, key=get_key)
            
            completed = False   # Indicates if a rule that has precedence over others was executed
            no_change_time = 0   # If change==True, no_change_time will represent the length of a sequence of False, and vice-versa 
            i = 0
            while ((i < n) and (len(types) > 0)):
                increment = 0   # When applying these rescoring rules, we have to consider positions that are beyond our current one, so
                                # we won't need to consider them again, this variable will be used to skip them
                
                if (current[i] != change):
                    no_change_time += 1   # Adds up until we reach the sequence's end
                else:
                    if (no_change_time < min(thresh_values)):   # This is the case when no rules are appliable
                        no_change_time = 0   # So we just prepare for a new count
                    else:
                        # First, we'll try type 1 rules, because they are "stronger"
                        # We'll start by testing the rule with the biggest threshold
                        if (no_change_time >= type1[len(type1) - 1][0]): 
                            completed, increment = rule_type1(type1[len(type1) - 1][0], type1[len(type1) - 1][1], i)
                            if completed:
                                type1[len(type1) - 1][2] += 1   # Rule count updated
                            else:
                                # If we can't apply the above mentioned rule, let's try all others!
                                for j in range(len(type1) - 2, -1, -1):
                                    completed, increment = rule_type1(type1[j][0], type1[j][1], i)
                                    if completed:
                                        type1[j][2] += 1
                                        break
                                
                        else:
                            # This loop finds out the biggest threshold value adequate to our sequence length and tries all rules
                            # with smaller threshold values until one is completed or all have been tried
                            for j in range(len(type1) - 1):
                                if ((no_change_time >= type1[j][0]) and (no_change_time < type1[j + 1][0])):
                                    for k in range(j, -1, -1):
                                        completed, increment = rule_type1(type1[k][0], type1[k][1], i)
                                        if completed:
                                            type1[k][2] += 1
                                            break
                                    break
                        # If none of our type 1 rules were appliable, we'll try type 0 rules
                        if ~completed:
                            # Again we'll start by considering the biggest threshold possible
                            if (no_change_time >= type0[len(type0) - 1][0]):
                                increment = rule_type0(type0[len(type0) - 1][1],i)
                                type0[len(type0)-1][2] += 1
                            else:
                                # Then we'll consider the smaller until we find the adequate one
                                for j in range(len(type0)-1):
                                    if ((no_change_time >= type0[j][0]) and (no_change_time < type0[j+1][0])):
                                        increment = rule_type0(type0[j][1], i)
                                        type0[j][2] += 1
                                        break
                                    
                        no_change_time = 0   # Prepares for a new count
                i += 1 + increment
                
            rule_count = []
            if (len(types) > 0):   # Rules were given
                if (len(type0) > 0):
                    if (len(type1) > 0):   # Both rule types given
                        rule_count = [transpose(type0)[2], transpose(type1)[2]]
                    else:   # Only type 0 rules
                        rule_count = transpose(type0)[2]
                else:
                    if (len(type1) > 0):   # Only type 1 rules
                        rule_count = transpose(type1)[2]
                        
            return current, rule_count
        
        n = len(self.activity)

        weighted = self.activity.copy()
        rescored = self.activity.copy()
        
        # Weighted sum
        b = len(self.weights_before) - 1
        a = len(self.weights_after)
        
        if n > (b+a+1):
            weighted[0:(b + 1)] = [(self.P*(dot(self.activity[i::-1], self.weights_before[(b - i):(b + 1)]) + dot(self.activity[(i + 1):(i + 1 + a)], self.weights_after))) for i in range(b + 1)]
            weighted[b:(n - a)] = [(self.P*(dot(self.activity[(i - b):(i + 1)], self.weights_before) + dot(self.activity[(i + 1):(i + 1 + a)], self.weights_after))) for i in range(b, (n - a))]
            weighted[(n - a):n] = [(self.P*(dot(self.activity[(i - b):(i + 1)], self.weights_before) + dot(self.activity[i:n], self.weights_after[0:(n - i)]))) for i in range((n - a), n)]
        else:
            weighted = zeros(n)

        filtered_weighted = weighted.copy()
        
        # Rescoring
        rule_count = [[],[]]
        rescored_weighted,rule_count[0] = rescoring_rules(where(weighted >= 1,True,False),True,self.types,self.max_values,self.thresh_values)
        rescored,rule_count[1] = rescoring_rules(where(rescored >= 1,True,False),True,self.types,self.max_values,self.thresh_values)
        self.rule_count = rule_count 
        
        for i in range(n):
            if rescored_weighted[i] < 1: # Positions containing keep_state will receive the state present in the array previouss
                rescored_weighted[i] = previous[i]
            else: # Positions containing change_state will receive the new state
                rescored_weighted[i] = self.new_state  
                
            if rescored[i] < 1:
                rescored[i] = previous[i]
            else:
                rescored[i] = self.new_state

            if filtered_weighted[i] < 1:
                filtered_weighted[i] = previous[i]
            else:
                filtered_weighted[i] = self.new_state
        
        self.filtered_weighted = filtered_weighted # Filtered weighted
        self.weighted = weighted # Weighted sum
        self.rescored = rescored # Rescored 
        self.rescored_weighted = rescored_weighted # Weighted and rescored
        
    def print_counts(self):
        print(self.rule_count)
    
    def plot_weighted(self, f=1):   # f == 1 for filtered results
        if f == 1: 
            plt.figure()
            plt.plot(self.weighted)
            plt.xlabel('Time [min]')
            plt.ylabel('Weighted sum results')
            plt.ylim(-0.5,1.5)
            plt.show()   
        else: 
            plt.figure()
            plt.plot(self.unfiltered_weighted)
            plt.xlabel('Time [min]')
            plt.ylabel('Weighted sum unfiltered output')
            plt.show()    
    
    def plot_output(self): 
        plt.figure()
        plt.plot(self.rescored_weighted)
        plt.xlabel('Time [min]')
        plt.ylabel('Output states')
        plt.ylim(-0.5,2.5)
        plt.show()
    
    def plot_rescored(self): 
        plt.figure()
        plt.plot(self.rescored)
        plt.xlabel('Time [min]')
        plt.ylabel('Rescoring results')
        plt.ylim(-0.5,1.5)
        plt.show()
            
    def plot_data(self): 
        plt.figure()
        plt.plot(self.activity)
        plt.xlabel('Time [min]')
        plt.ylabel('Activity data')
        plt.show()