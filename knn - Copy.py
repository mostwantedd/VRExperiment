

import glob
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn import metrics


# change this path to your own directory
directoryPath='C:/Users/CAMERA/Desktop/allppts/'

#the column headers
names = ['Participant', 'Seconds', 'DataIsValid', 'Date', 'Time', 'DirectionX', 'DirectionY', 'DirectionZ', 'OriginX', 'OriginY', 'OriginZ', 'Blink', 'StimuliName', 'IsLookingAtCanvas', 'TaskType', 'SeshNo']
num_ppts = 7
num_sessions = 3
num_stimuli = 100
timepoints = 800 # this is per session #change to 800
datapoints = 3 # xyz direction, xyz origin

# takes in origin and direction coordinates as one list, and converts this into a list of 3 gazepoint coordinates
def calculate_gazepoint(data):
    #Define plane
    distance = data[0:3]
    origin = data[3:]
    planeNormal = np.array([0, 0, 1]) 
    planePoint = np.array([0, 1, 0.75]) #Any point on the plane of the image
    #print("dist and origin ", distance, origin)
    gazePoint = []
    
    # get intersection of gaze ray with the image plain stored in gazePoint
    rayDirection = np.array([distance[0], distance[1], distance[2]])
    rayPoint = np.array([origin[0], origin[1], origin[2]])
    ndotu = planeNormal.dot(rayDirection) 
    epsilon = 1e-6 
    if abs(ndotu) < epsilon:
            print ("no intersection or line is within plane")
    
    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    Psi = Psi.tolist()
    #print(Psi, type(Psi))
    gazePoint.append(Psi)
    #print(" gaze point ", gazePoint[0])
    return gazePoint[0]

# creating the arrays for storing the data in ready for use in classification algorithms
# image ID by time points by gaze data for each session
# for example would be 3, 100, 500, 3
all_images_session1 = np.zeros(num_ppts*num_stimuli *timepoints *datapoints)
all_images_session1 = all_images_session1.reshape(num_ppts,num_stimuli, timepoints, datapoints)
all_images_session2 = np.zeros(num_ppts*num_stimuli *timepoints *datapoints)
all_images_session2 = all_images_session2.reshape(num_ppts,num_stimuli, timepoints, datapoints)
all_images_session3 = np.zeros(num_ppts*num_stimuli *timepoints *datapoints)
all_images_session3 = all_images_session3.reshape(num_ppts,num_stimuli, timepoints, datapoints)

print(all_images_session1.shape)


#create a dictionary of ID names as keys and target int as value,
# for example ' i05june05_static_street_boston_p1010855' may have target value as '99'
# this is so the target ints can be mapped back to the imageID
# any ppt number could be used here
target_list = list(range(0,num_stimuli))
IDs = []
for filename in glob.glob(directoryPath+'0241/*.csv'):#0241
    if 'Encoding' in filename:
        image_ID =  filename.split("Encoding_",1)[1]
        image_ID = image_ID.split('.csv')[0]
        #print(image_ID)
        IDs.append(image_ID)
   
image_to_target_dict = dict(zip(IDs, target_list))
#print("dictionary is ", image_to_target_dict)


"""
Extracting the data from the csv files, originally as a dataframe.
Find the stimuliname for each csv file, and get the corresponding target int, e.g. '99'
Sessions are initially processed separately, so that session 3 can be discarded if it impairs classification performance
Iterates through, first by ppts, then by rows in the csv files (timepoints)
For each time point the gazepoint is calculated and stored in on of the all_images_sessions arrays
"""
dirp = directoryPath+'*/'
print(dirp)
for ppt_count, ppt in enumerate(glob.glob(dirp)):
    print("ppt number: ",ppt_count)
    for filename in glob.glob(ppt+'*.csv'):
        file_contents = pd.read_csv(filename,  names=names, header=None)
        
        if 'Recall' in filename: #change to Encoding/Recall
            
            target = file_contents['StimuliName'][0]
            target = target.replace(" ", "")
            #print("image ID ", target)
            targetImage = image_to_target_dict.get(target)  #######
            #print(targetImage)
            s1_index = 0
            s2_index = 0
            s3_index = 0
            
            for index, timepoint_row in file_contents.iterrows():
                if timepoint_row['SeshNo'] == ' S1':# and timepoint_row['DataIsValid'] == ' True':
                    if(s1_index < timepoints):                        
                        gazePoint = calculate_gazepoint(timepoint_row.tolist()[5:11])
                      #  if timepoint_row['DataIsValid'] == ' True':
                        all_images_session1[ppt_count][targetImage][s1_index] =  gazePoint
                        s1_index +=1
                if timepoint_row['SeshNo'] == ' S2':# and timepoint_row['DataIsValid'] == ' True':
                    if(s2_index < timepoints):
                        gazePoint = calculate_gazepoint(timepoint_row.tolist()[5:11])
                       # if timepoint_row['DataIsValid'] == ' True':
                        all_images_session2[ppt_count][targetImage][s2_index] =  gazePoint
                        s2_index +=1
                if timepoint_row['SeshNo'] == ' S3':# and timepoint_row['DataIsValid'] == ' True':
                    if(s3_index < timepoints):
                        gazePoint = calculate_gazepoint(timepoint_row.tolist()[5:11])
                      #  if timepoint_row['DataIsValid'] == ' True':
                        all_images_session3[ppt_count][targetImage][s3_index] =  gazePoint
                        listt = all_images_session3[ppt_count][targetImage][s3_index]
                        s3_index +=1 
                        

    

"""
Preparing the data into correct shape arrays, and concatenating the sessiosn together.
Also creating a list of target labels, essentially [0-99,0-99,0-99] (as three sessions)
"""
all_session = np.concatenate((all_images_session1, all_images_session2, all_images_session3), axis = 1)
across_ppts_all_session = all_session.reshape(-1,timepoints, datapoints)
#create labels, 0-99, repeated by number of sessions and ppts
labels = list(range(0,num_stimuli)) * num_sessions * num_ppts
# should perhaps look into scaling the data first
# must be in 2D format for this regression, so the last two dimensions are collapsed
# timepoints * datapoints (gaze_data = 3)
across_ppts_all_LR = across_ppts_all_session.reshape(num_ppts * num_sessions * num_stimuli, timepoints*datapoints)
eye_data = across_ppts_all_LR # will be a np array
Image_labels = labels # will also be an np array
X_train, X_test, y_train, y_test = train_test_split(eye_data , Image_labels , test_size=0.2, random_state=8, shuffle=True)
#y_train = np.asarray(y_train)
#y_test = np.asarray(y_test)

"""
print("Now carrying out Random Forest Classification")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
#rr = [1300, 1500, 200, 300, 400, 1800, 600, 1000, 10000]
#for r in rr:
#clf=RandomForestClassifier(n_estimators = 500)#r
forest = RandomForestClassifier(random_state = 1)
modelF = forest.fit(X_train, y_train)
y_predF = modelF.predict(X_test)
n_estimators = [100, 300, 500, 800, 1200]
max_depth = [5, 8, 15, 25, 30]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10] 
hyperF = dict(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf)
gridF = GridSearchCV(forest, hyperF, cv = 3, verbose = 1, n_jobs = -1)
bestF = gridF.fit(X_train, y_train)
forestOpt = RandomForestClassifier(random_state = 1, max_depth = 15, n_estimators = 500, min_samples_split = 2, min_samples_leaf = 1)                                   
modelOpt = forestOpt.fit(X_train, y_train)
y_pred = modelOpt.predict(X_test)
"""

"""
print("Now carrying out KNN")
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=50)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


print("Now carrying out Random Forest Classification")
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators = 500)#r 1000
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)  
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Acc: 0.047, 0.092, 0.121, 0.109, 0.111, 0.130, 0.121, 0.121, 0.119, 0.119


print("Now carrying out Logistic Regression")
lr = LogisticRegression(penalty ='l2', random_state=6, solver ='lbfgs', max_iter=2000, multi_class = 'auto')
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
score = lr.score(X_test, y_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred)) # chance

print("Now carrying out SVM")
from sklearn import svm
clf = svm.SVC(kernel = 'rbf', gamma = 1e-1, C=1e2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


print("Now carrying out LSTM")
X_train, X_test, y_train, y_test = train_test_split(across_ppts_all_session , labels , test_size=0.2, random_state=8, shuffle=True)
seq_len = 800
model = Sequential()
model.add(LSTM(256, input_shape=(seq_len, datapoints)))
model.add(Dense(1, activation='sigmoid'))
### need to convert data types into arrays if there are any lists
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
adam = Adam(lr=0.95)#0.00001
#chk = ModelCheckpoint('best_model.pkl', monitor='val_acc', save_best_only=True, mode='max', verbose=1)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.fit(X_train, y_train, epochs=80, batch_size= 30,  validation_data=(X_test,y_test))
#print(model.parameters)
test_preds = model.predict(X_test)
accuracy = accuracy_score(y_test, test_preds)
print(accuracy)
"""



"""
C_2d_range = [1e-2, 1, 1e2]
gamma_2d_range = [1e-1, 1, 1e1]
classifiers = []
for C in C_2d_range:
    for gamma in gamma_2d_range:
        clf = svm.SVC(C=C, gamma=gamma)
        clf.fit(X_train, y_train)
        classifiers.append((C, gamma, clf))
        y_pred = clf.predict(X_test)
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
 
Acc: 0.0033, 0.0033, 0.0016, 0.0133, 0.0033, 0.0033, 0.0166, 0.0033, 0.0033    
"""
 

"""
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state = 42)
from pprint import pprint# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())
from sklearn.model_selection import RandomizedSearchCV# Number of trees in random forest
n_estimators = [1300, 1500, 200, 300, 400, 1800, 600, 1000, 10000]
# Number of features to consider at every split
max_features = ['auto', 'sqrt', 'log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]# Create the random grid

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)# Fit the random search model
rf_random.fit(X_train, y_train)
rf_random.best_params_
"""

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from scipy.stats import sem
from numpy import mean
from matplotlib import pyplot

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm


# evaluate a model with a given number of repeats
def evaluate_model(X, y, repeats):
	# prepare the cross-validation procedure
	cv = RepeatedKFold(n_splits=10, n_repeats=repeats, random_state=1)
	# create model
	model = LogisticRegression(penalty ='l2', random_state=6, solver ='lbfgs', max_iter=2000, multi_class = 'auto')
    #model = svm.SVC(kernel = 'rbf', gamma = 1e-1, C=1e2)
    #model = LogisticRegression(penalty ='l2', random_state=6, solver ='lbfgs', max_iter=2000, multi_class = 'auto')
    #model = RandomForestClassifier(n_estimators = 500)
    #model = KNeighborsClassifier(n_neighbors=50)
	# evaluate model
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores


repeats = range(1, 6)
results = list()
for r in repeats:
	# evaluate using a given number of repeats
	scores = evaluate_model(eye_data, Image_labels, r)
	# summarize
	print('>%d mean=%.4f se=%.3f' % (r, mean(scores), sem(scores)))
	# store
	results.append(scores)
# plot the results
pyplot.boxplot(results, labels=[str(r) for r in repeats], showmeans=True)
pyplot.show()

