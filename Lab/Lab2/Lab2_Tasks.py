'''
COMP 2211 Exploring Artificial Intelligence
Lab 2 Naive Bayes Classifier

Name: YIP, Pak To Paco    
SID: 20771007
'''

import numpy as np

train = np.loadtxt("heart_disease_train_dataset.csv", delimiter=',', skiprows=1)
test = np.loadtxt("heart_disease_test_dataset.csv", delimiter=',', skiprows=1)

NO = 0
YES = 1

train_features = train[:, :-1] # Evidence)
train_labels = train[:, -1]    # Belifes)

### 1. Find Relative Frequencies

num_heart_disease_yes = 0 # Count of heart_disease = yes
num_heart_disease_no = 0  # Count of heart_disease = no

freq = np.zeros((2, 6, 3))      #Given yes/no, 6 evidence ,up to 3 value

def find_freq(B, row):
  for j, value in enumerate(row):
    freq[B, j, int(value)] += 1
  
for i, row in enumerate(train_features):
  if train_labels[i] == 1:
    num_heart_disease_yes += 1
    find_freq(YES, row)
    
  else:
    num_heart_disease_no += 1
    find_freq(NO, row)
  

### 2. Find the Probabilities

heart_disease_yes = num_heart_disease_yes/train.shape[0] # P(heart_disease = yes)
heart_disease_no = num_heart_disease_no/train.shape[0]   # P(heart_disease = no)

prob_of_evidence = np.zeros((2, 6, 3))

# Each count of evidence will be divide by num of yes/no
prob_of_evidence[0] = freq[0]/num_heart_disease_no
prob_of_evidence[1] = freq[1]/num_heart_disease_yes

### 3. Prediction

test_features = test[:, :-1] # All except the last column.
test_labels = test[:, -1]    # Only the last column.
predict_labels = np.zeros_like(test_labels)

def sum_of_log(B, row):
  sum = 0
  for j, value in enumerate(row):
    prob_reqd = prob_of_evidence[B, j, int(value)]
    sum += np.log(prob_reqd)
  return sum

log_heart_disease_yes = np.log(heart_disease_yes) # log_e of P(heart_disease = yes)
log_heart_disease_no = np.log(heart_disease_no)   # log_e of P(heart_disease = no)
#TODO

for i, row in enumerate(test_features):
  
  predict_yes = log_heart_disease_yes + sum_of_log (YES, row)# log_e of P(heart_disease = yes)
  predict_no = log_heart_disease_no + sum_of_log (NO, row)   # log_e of P(heart_disease = no)
  
  # Find out max
  predict_labels[i] = YES
  if predict_no > predict_yes:
    predict_labels[i] = NO

### 4. Test Accuracy

num_match = 0
for i in range(predict_labels.shape[0]):
  if predict_labels[i] == test_labels[i]:
    num_match += 1
accuracy_score = num_match/predict_labels.shape[0]

print(accuracy_score)
print(predict_labels)
print(test_labels)