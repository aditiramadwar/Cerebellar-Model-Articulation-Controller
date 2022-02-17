from cgi import test
import numpy as np
import math
import matplotlib.pyplot as plt
import time
# Initialize variables that contribute to our model
num_weights = 35
# we will have 3 overlapping areas
gen_factor = 3
min_val = 0
max_val = 100
# Determine the number of association cells required for this particular model
assoc_dimensions = num_weights - gen_factor + 1

# Normalize the input data to create the active association cells for each input,
# this function gives out the index of in the range of 35 of where the active cells should start from for 
# each input. 
# This function creates a hashmap for input data and all the corresponding active cell indices
def get_association_map(data):
    association_matrix = {}
    for data_val in (data):
        activation_idx = (assoc_dimensions-2) * ((data_val - min_val)/(max_val - min_val)) + 1
        assoc_vec_ind = int(math.floor(activation_idx))
        association_matrix[data_val] = assoc_vec_ind
    return association_matrix

# Train the model
def train(train_data, des_data, epochs = 10000, learning = 0.1):
    weight_array = np.ones(num_weights)
    association_matrix = get_association_map(tran_data)
    start_time = time.time()
    for i in range(epochs):
        out = []
        for ind, data_val in enumerate(train_data):
            assoc_idx = association_matrix[data_val]
            y_output = np.sum(weight_array[assoc_idx : assoc_idx + gen_factor])
            error =  des_data[ind] - y_output
            out = np.append(out, y_output)
            correction = (learning * error)/gen_factor
            weight_array[assoc_idx : (assoc_idx + gen_factor)] = [(weight_array[ind] + correction) for ind in range(assoc_idx, (assoc_idx + gen_factor))]
        # check the accuracy of the predictions at each epoch
        accuracy = check_accuracy(out, des_data)
        # check thetotal time taken for computation after each epoch
        time_taken = time.time() - start_time
        # If accuracy is reached before running through all the epochs then that's the optimal solution and break out of the loop 
        # This checks for convergence
        if (accuracy*100 > 90):
            break
        # If total time taken is more than 5 seconds, break out of the loop
        if (time.time() - start_time > 5):
            break
    return weight_array, accuracy, time_taken

# Test the trained model
def test(test_data, test_res_des, weights):
    out = np.array([])
    test_assoc = get_association_map(test_data)
    for data_val in (test_data):
            assoc_idx = test_assoc[data_val]
            out = np.append(out, np.sum(weights[assoc_idx : assoc_idx + gen_factor]))
    accuracy = check_accuracy(out, test_res_des)
    return out, accuracy

# Check accuracy between desired data and predicted data from our discrete model
def check_accuracy(output, desired, err_th = 200):
    result = np.count_nonzero(abs(output-desired) < err_th)
    accuracy = result/len(output)
    return accuracy

######################## main execution ########################
data = np.array([])
for i in range (min_val, max_val):
    data = np.append(data, i)
# Randomly shuffle the data
np.random.shuffle(data)
# Testing the model for y = x^2
des_data = data**2

# Divide the data into training set and testing set that can be later used
tran_data = data[:70]
tran_data_des = des_data[:70]
test_data = data[70:]
test_res_des = des_data[70:]

# Get the weights by training the data
weight_array, accuracy, time_taken = train(tran_data, tran_data_des)
print("Training data accuracy:", np.round(accuracy*100, 2), "%, Trained in",time_taken,"seconds")

# Test out model using the testing setand the derived weightage
test_out, accuracy = test(test_data, test_res_des, weight_array)
print("Testing data accuracy:", np.round(accuracy*100, 2), "%")

## plot the desired and the predicted data ##
plt.plot(np.sort(test_data), np.sort(test_res_des), 'g', label='Desired Output')
plt.plot(np.sort(test_data), np.sort(test_out), 'r', label='Predicted Output')

plt.legend(loc="lower right")
plt.xlabel('x')
plt.ylabel('x^2')
plt.savefig('result_discrete.png')
plt.show()