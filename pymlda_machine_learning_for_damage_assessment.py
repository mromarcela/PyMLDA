"""PyMLDA - Machine Learning for Damage Assessment


**Authors:** Jefferson Coelho, Marcela Machado, Amanda Aryda \\
**Software version:** V.1 \\
**Tutorial description:** https://github.com/mromarcela/PyMLDA


The PyMLDA software uses additional packages required for the algorithm's operation. Therefore before the code excuttion make sure you have the followinf packages installed.

Here is a breif tutorial on how to create a virtual environment,  in case you wish, without affecting other projects on your machine. Please see below:
1. Open the comand prompt and run one at a time the following the steps 2 to 12.
2. conda create -n Teste_ML
3. conda activate Teste_ML
4. pip install openpyxl
5. pip install numpy
6. pip install matplotlib
7. pip install pandas
8. pip install seaborn
9. pip install imbalanced-learn
10. pip install -U scikit-learn
11. pip install xgboost
12. Run the PyMLDA (code).
"""

#====== Import packages need to run the code - Pandas, Numpy, matplotlib, Seaborn, warnings, imblearn, sklearn and xgboost.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVR
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore') # Ignore warnings

#====== Load the Index return sheet from the Excel file into a pandas dataframe

# Load the Excel file into a DataFrame:
dataset= pd.read_excel('DI_FRAC_Exp-estimation.xlsx', sheet_name='DI_FRAC')

# Remove the 'Mass loss [%]' and 'Multiclass_classification' columns
dataset.drop(['Mass loss [%]','Multiclass classification'], axis=1, inplace=True)

"""# **Unsupervised**"""

# Print header of DataFrame
dataset.head(10)

# Using the elbow method to find the optimal number of clusters
wcss = []  # Within-Cluster Sum of Square
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(dataset)
    # inertia method returns wcss for that model
    wcss.append(kmeans.inertia_)

# Plot the graph of k versus WCSS
plt.plot(range(1,11),wcss, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion (Within Cluster Sum of Squared Errors)')
plt.title('The Elbow Method showing the optimal k')
plt.show()

# Build the Kmeans clustering model

k = 4  # k: number of clusters
kmeans = KMeans(n_clusters = k, init = 'k-means++', random_state = 0) # Create kemans model
y_kmeans_pred = kmeans.fit_predict(dataset) # Fitting K-Means to the dataset

# Returns the clustering labels

labels=kmeans.labels_
labels

# Cluster data using the created model
grupos = kmeans.predict(dataset)

# Create cluster feature
dataset=pd.DataFrame(dataset,columns=dataset.columns[0:3])
dataset['Clusters']=labels

#====== Plot the cluster with data points

# plt.scatter() function
plt.figure()
plt.scatter(dataset.values[grupos == 3, 0], dataset.values[grupos == 3, 1], s = 100, c = 'tab:blue', edgecolor='white', label = 'Healthy')
plt.scatter(dataset.values[grupos == 1, 0], dataset.values[grupos == 1, 1], s = 100, c = 'tab:orange', edgecolor='white', label = '2,96% Damage')
plt.scatter(dataset.values[grupos == 0, 0], dataset.values[grupos == 0, 1], s = 100, c = 'tab:green', edgecolor='white', label = '5,92% Damage') #edgecolor='black'
plt.scatter(dataset.values[grupos == 2, 0], dataset.values[grupos == 2, 1], s = 100, c = 'tab:red', edgecolor='white', label = '8,87% Damage')

# Change the appearance of ticks and tick labels.
plt.minorticks_on()
plt.tick_params(direction='in',right=True, top=True)
plt.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=True)
plt.tick_params(direction='in',which='minor', length=5, bottom=True, top=True, left=True, right=True)
plt.tick_params(direction='in',which='major', length=10, bottom=True, top=True, left=True, right=True)
plt.xlabel('$DI_2$', fontsize =12 )
plt.ylabel('$DI_1$', fontsize = 12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Function add a legend
plt.legend(fontsize=12)
plt.legend(loc='upper left',fontsize=12)

# plt.grid()
plt.legend(scatterpoints=1)

# Save figure in PDF
plt.savefig('dados_exp.pdf', format='pdf', transparent=True, dpi=1200, bbox_inches='tight')

# function to show the plot
plt.show()

# Visualising of New DataFrame with clusters
dataset

"""# **Supervised**

# Classification
"""

# Checking data imbalance after clustering

print(dataset['Clusters'].unique(),'\n') # labels da classe - variável categórica

features = dataset.Clusters.value_counts()
print(features)
plt.figure()
features.plot(kind='bar', title='Count (Clusters)');

# convert categorical columns into numerical

labelEncoder = LabelEncoder()
labelEncoder.fit(dataset['Clusters'])
dataset['Clusters'] = labelEncoder.transform(dataset['Clusters'])

dataset

# split the data into inputs (x) and outputs (y)
X = dataset.iloc[:, :2] # inputs
y = dataset.iloc[:, 2]  # outputs

# split the data into training and testing sets ()
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.20, random_state=42)

# Applying the undersampling technique to balance samples

rus = RandomUnderSampler()
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

print(y_train.shape)
print(y_train.value_counts(), '\n')

print(y_train_rus.shape)
print(y_train_rus.value_counts(),'\n')

"""# SVM"""

# Train the SVM model on the entire training set

SVM_OvO = svm.SVC(kernel='linear', C=100, decision_function_shape='ovo') # Create SVM model - One vs One
SVM_OvO.fit(X_train_rus, y_train_rus) # fit classifier to training set

# Predict the samples for testing set

y_pred= SVM_OvO.predict(X_test)

print('Classificador:', y_pred[1])
print('Valor Real:', y_test.iloc[1])

#===== Metrics to Evaluate your Classification Model SVM

# Compute and print classification metrics (Accuracy, F1-Score, Precision and Recall)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('\nF1-score:', f1_score(y_test, y_pred, average='micro'))
print('\nPrecision: %.3f' % precision_score(y_test, y_pred,average='micro'))
print('\nRecall: %.3f' % recall_score(y_test, y_pred,average='micro'))

# Print classification report for model
print('\nClassification report:\n')
print(classification_report(y_test, y_pred))

# Cross-validation for classification

scores=cross_val_score(SVM_OvO, X, y, cv=5, scoring='accuracy')

# Print cross-validation
print("Cross-validation scores: {}". format(scores))
print("Average cross-validation score: {}". format(scores.mean()))

# Creating a Confusion Matrix

cm_SVM = confusion_matrix(y_test, y_pred)
cm_SVM

# Plot Confusion Matrix SVM

plt.figure(figsize=(8,6))

group_counts = ["{0:0.0f}".format(value) for value in
                cm_SVM.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     cm_SVM.flatten()/np.sum(cm_SVM)]

labels = [f"{v1}\n{v2}\n" for v1, v2 in
          zip(group_counts,group_percentages)]
labels = np.asarray(labels).reshape(len(cm_SVM),len(cm_SVM))

ax = sns.heatmap(cm_SVM, annot=labels, fmt='', cmap=plt.cm.Reds)
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');

ax.xaxis.set_ticklabels(['8,87% Damage', '5,92% Damage', '2,96% Damage', 'Healthy']);
ax.yaxis.set_ticklabels(['8,87% Damage', '5,92% Damage', '2,96% Damage', 'Healthy'])

# Save the plot with dpi=1200 in 'pdf'
plt.savefig('svm_cm_.pdf', format='pdf', transparent=True, dpi=1200, bbox_inches='tight' )

plt.show()

"""# KNN"""

# Train the KNN model on the entire training set

knn = KNeighborsClassifier(n_neighbors = 3, metric= 'euclidean') # Create KNN model
knn.fit(X_train_rus,y_train_rus) #fit classifier to training set

# Predict the samples for testing set

y_pred = knn.predict(X_test)

print('Classificador:', y_pred[1])
print('Valor Real:', y_test.iloc[1])

#===== Metrics to Evaluate your Classification Model

# Compute and print classification metrics (Accuracy, F1-Score, Precision and Recall)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('\nF1-score:', f1_score(y_test, y_pred, average='micro'))
print('\nPrecision: %.3f' % precision_score(y_test, y_pred,average='micro'))
print('\nRecall: %.3f' % recall_score(y_test, y_pred,average='micro'))

# Print classification report for model
print('\nClassification report:\n')
print(classification_report(y_test, y_pred))

# Cross-validation for classification SVM

scores=cross_val_score(knn, X, y, cv=5, scoring='accuracy')

# Print cross-validation
print("Cross-validation scores: {}". format(scores))
print("Average cross-validation score: {}". format(scores.mean()))

# Creating a Confusion Matrix

cm_knn = confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix KNN

plt.figure(figsize=(8,6))
group_counts = ["{0:0.0f}".format(value) for value in
                cm_knn.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     cm_knn.flatten()/np.sum(cm_knn)]

labels = [f"{v1}\n{v2}\n" for v1, v2 in
          zip(group_counts,group_percentages)]
labels = np.asarray(labels).reshape(len(cm_knn),len(cm_knn))

ax = sns.heatmap(cm_knn, annot=labels, fmt='', cmap=plt.cm.Blues)
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');

ax.xaxis.set_ticklabels(['8,87% Damage', '5,92% Damage', '2,96% Damage', 'Healthy']);
ax.yaxis.set_ticklabels(['8,87% Damage', '5,92% Damage', '2,96% Damage', 'Healthy'])

# Save the plot with dpi=1200 in 'pdf'
plt.savefig('knn_cm_.pdf', format='pdf', transparent=True, dpi=1200, bbox_inches='tight' )
plt.show()

"""# Naive Bayes"""

# Train the Naive Bayes model on the entire training set
gnb =  GaussianNB() # Create Naive Bayes model
gnb.fit(X_train_rus, y_train_rus) # fit classifier to training set

# Predict the samples for testing set

y_pred = gnb.predict(X_test)

print('Classificador:', y_pred[1])
print('Valor Real:', y_test.iloc[1])

#===== Metrics to Evaluate your Classification Model

# Compute and print classification metrics (Accuracy, F1-Score, Precision and Recall)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('\nF1-score:', f1_score(y_test, y_pred, average='micro'))
print('\nPrecision: %.3f' % precision_score(y_test, y_pred,average='micro'))
print('\nRecall: %.3f' % recall_score(y_test, y_pred,average='micro'))

# Printing classification Report
print('\nClassification report:\n')
print(classification_report(y_test, y_pred))

# Cross valcross_val_score

scores=cross_val_score(gnb, X, y, cv=5, scoring='accuracy')

# Print cross-validation
print("Cross-validation scores: {}". format(scores))
print("Average cross-validation score: {}". format(scores.mean()))

# Creating a Confusion Matrix

cm_gnb = confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix Naive Bayes

plt.figure(figsize=(8,6))

group_counts = ["{0:0.0f}".format(value) for value in
                cm_gnb.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     cm_gnb.flatten()/np.sum(cm_gnb)]

labels = [f"{v1}\n{v2}\n" for v1, v2 in
          zip(group_counts,group_percentages)]
labels = np.asarray(labels).reshape(len(cm_gnb),len(cm_gnb))

ax = sns.heatmap(cm_gnb, annot=labels, fmt='', cmap=plt.cm.Oranges)
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');

ax.xaxis.set_ticklabels(['8,87% Damage', '5,92% Damage', '2,96% Damage', 'Healthy']);
ax.yaxis.set_ticklabels(['8,87% Damage', '5,92% Damage', '2,96% Damage', 'Healthy'])

#Save the plot with dpi=1200 in 'pdf'
plt.savefig('nb_cm_.pdf', format='pdf', transparent=True, dpi=1200, bbox_inches='tight' )
plt.show()

"""# Random Forest"""

# Train the Random forest model on the entire training set

rf = RandomForestClassifier(criterion='gini') # Create Random forest model
rf.fit(X_train_rus, y_train_rus) # fit classifier to training set

# Predict the samples for testing set

y_pred = rf.predict(X_test)

print('Classificador:', y_pred[1])
print('Valor Real:', y_test.iloc[1])

#===== Metrics to Evaluate your Classification Model

# Compute and print classification metrics (Accuracy, F1-Score, Precision and Recall)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('\nF1-score:', f1_score(y_test, y_pred, average='micro'))
print('\nPrecision: %.3f' % precision_score(y_test, y_pred,average='micro'))
print('\nRecall: %.3f' % recall_score(y_test, y_pred,average='micro'))

# Print classification report for model
print('\nClassification report:\n')
print(classification_report(y_test, y_pred))

# crovss validation

scores=cross_val_score(rf, X, y, cv=5, scoring='accuracy')

# Print cross-validation
print("Cross-validation scores: {}". format(scores))
print("Average cross-validation score: {}". format(scores.mean()))

# Creating a Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix Random Forest

plt.figure(figsize=(8,6))

group_counts = ["{0:0.0f}".format(value) for value in
                cm_rf.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     cm_rf.flatten()/np.sum(cm_rf)]

labels = [f"{v1}\n{v2}\n" for v1, v2 in
          zip(group_counts,group_percentages)]
labels = np.asarray(labels).reshape(len(cm_rf),len(cm_rf))

ax = sns.heatmap(cm_rf, annot=labels, fmt='', cmap=plt.cm.Greens)
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');

ax.xaxis.set_ticklabels(['8,87% Damage', '5,92% Damage', '2,96% Damage', 'Healthy']);
ax.yaxis.set_ticklabels(['8,87% Damage', '5,92% Damage', '2,96% Damage', 'Healthy'])

# Save the plot with dpi=1200 in 'pdf'
plt.savefig('rf_cm_.pdf', format='pdf', transparent=True, dpi=1200, bbox_inches='tight' )
plt.show()

"""# Decision Tree"""

# Train the Decision Tree model on the entire training set

dtree = DecisionTreeClassifier(criterion='gini')# Create decision Tree model
dtree.fit(X_train_rus, y_train_rus) # fit classifier to training set

# Predict the samples for testing set

y_pred = dtree.predict(X_test)

print('Classificador:', y_pred[1])
print('Valor Real:', y_test.iloc[1])

#===== Metrics to Evaluate your Classification Model

# Compute and print classification metrics (Accuracy, F1-Score, Precision and Recall)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('\nF1-score:', f1_score(y_test, y_pred, average='micro'))
print('\nPrecision: %.3f' % precision_score(y_test, y_pred,average='micro'))
print('\nRecall: %.3f' % recall_score(y_test, y_pred,average='micro'))

# Print classification report for model
print('\nClassification report:\n')
print(classification_report(y_test, y_pred))

# Cross-validation for classification

scores=cross_val_score(dtree, X, y, cv=5, scoring='accuracy')

# Print cross-validation
print("Cross-validation scores: {}". format(scores))
print("Average cross-validation score: {}". format(scores.mean()))

# Creating a Confusion Matrix

cm_dtree = confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix Decision Tree

plt.figure(figsize=(8,6))

group_counts = ["{0:0.0f}".format(value) for value in
                cm_dtree.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     cm_dtree.flatten()/np.sum(cm_dtree)]

labels = [f"{v1}\n{v2}\n" for v1, v2 in
          zip(group_counts,group_percentages)]
labels = np.asarray(labels).reshape(len(cm_dtree),len(cm_dtree))

ax = sns.heatmap(cm_dtree, annot=labels, fmt='', cmap=plt.cm.Purples)
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');

ax.xaxis.set_ticklabels(['8,87% Damage', '5,92% Damage', '2,96% Damage', 'Healthy']);
ax.yaxis.set_ticklabels(['8,87% Damage', '5,92% Damage', '2,96% Damage', 'Healthy'])

# Save the plot with dpi=1200 in 'pdf'
plt.savefig('dt_cm_.pdf', format='pdf', transparent=True, dpi=1200, bbox_inches='tight' )
plt.show()

"""# XGBClassifier"""

# Train the XGBoost model on the entire training set

xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_classes=9) # Create XGBoost model
xgb_model.fit(X_train_rus, y_train_rus) # fit classifier to training set

# Predict the samples for testing set

y_pred = xgb_model.predict(X_test)

print('Classificador:', y_pred[1])
print('Valor Real:', y_test.iloc[1])

#===== Metrics to Evaluate your Classification Model

# Compute and print classification metrics (Accuracy, F1-Score, Precision and Recall)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('\nF1-score:', f1_score(y_test, y_pred, average='micro'))
print('\nPrecision: %.3f' % precision_score(y_test, y_pred,average='micro'))
print('\nRecall: %.3f' % recall_score(y_test, y_pred,average='micro'))

# Print classification report for model
print('\nClassification report:\n')
print(classification_report(y_test, y_pred))

# Cross-validation for classification

scores=cross_val_score(xgb_model, X, y, cv=5, scoring='accuracy')

# Print cross-validation
print("Cross-validation scores: {}". format(scores))
print("Average cross-validation score: {}". format(scores.mean()))

# Creating a Confusion Matrix

cm_xgb = confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix XGBoost

plt.figure(figsize=(8,6))

group_counts = ["{0:0.0f}".format(value) for value in
                cm_xgb.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     cm_xgb.flatten()/np.sum(cm_xgb)]

labels = [f"{v1}\n{v2}\n" for v1, v2 in
          zip(group_counts,group_percentages)]
labels = np.asarray(labels).reshape(len(cm_xgb),len(cm_xgb))


ax = sns.heatmap(cm_xgb, annot=labels, fmt='',cmap= plt.cm.Blues)
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');

ax.xaxis.set_ticklabels(['8,87% Damage', '5,92% Damage', '2,96% Damage', 'Healthy']);
ax.yaxis.set_ticklabels(['8,87% Damage', '5,92% Damage', '2,96% Damage', 'Healthy'])

# Save the plot with dpi=1200 in 'pdf'
plt.savefig('xgb_cm_.pdf', format='pdf', transparent=True, dpi=1200, bbox_inches='tight' )
plt.show()

"""# **Regression**"""

# Sorts the DataFrame by the specified label.
dataset = dataset.sort_values(by="Clusters", ascending=False)

# split the data into inputs (x) and outputs (y)
X = dataset.iloc[:,0].values
y = dataset.iloc[:,2].values

#===== Data standardization
# Scale the data using StandardScaler
X = X.reshape(len(X),1)
y = y.reshape(len(y),1)

# Preprocessing using zero mean and unit variance scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

#===== Fitting SVR to the dataset
svm_reg = SVR(kernel='linear',gamma='scale', C=10.0, epsilon=0.1) # Create SVR model
svm_reg.fit(X, y) # Fitting SVR model

# Evaluating model using cross-validation and the mean squared error (MSE) and R-squared metrics
seed=50
np.random.seed(seed)
y_norm_Ravel=y.ravel()

scores_MSE=cross_val_score(svm_reg, X,y_norm_Ravel,cv=5,scoring='neg_mean_squared_error')
print("MSE_ Cross-validation scores: {}". format(scores_MSE))
print("Average Kfold cross-validation MSE_score: {}".format(scores_MSE.mean()))

scores_R2=cross_val_score(svm_reg, X, y_norm_Ravel,cv=5,scoring='r2')
print("\nR2_Cross-validation scores: {}". format(scores_R2))
print("Average R2_Cross-validation scores: {}".format(scores_R2.mean()))

#===== Data standardization for new data
X2 = dataset.iloc[:,2].values
X2 = X2.reshape(len(X2),1)

sc_X2 = StandardScaler()
X2 = sc_X2.fit_transform(X)

# Predicting a new result
y_test_pred = svm_reg.predict(sc_X2.transform(X2))

# Convert y_test_pred to 2D
y_test_pred = y_test_pred.reshape(-1,1)

# Taking the inverse of the scaled value
y_test_pred_inv = sc_y.inverse_transform(y_test_pred)

# Create a simple Pandas DataFrame
y_pred_damage = pd.DataFrame(y_test_pred_inv, columns = ['Damage_pred'])

# Taking the inverse of the scaled value
y_real_inv = sc_y.inverse_transform(y)

# Data standardization - inverse transform
y_real_damage = pd.DataFrame(sc_y.inverse_transform(y), columns = ['Damage_real']).astype(int)

# Combining dataFrame real e predict.
data_result = pd.merge(y_real_damage, y_pred_damage, left_index = True, right_index = True, how = "inner")

# Compute and Printing the mean
mean_damage_prev = data_result.groupby('Damage_real')[['Damage_pred']].mean().sort_values(by='Damage_real',ascending=True).reset_index()
print(mean_damage_prev)

# Compute and Printing the standard deviation
Std_damage_prev = data_result.groupby('Damage_real')[['Damage_pred']].std().sort_values(by='Damage_real',ascending=True).reset_index()
print(Std_damage_prev)

#===== Plot predicted values vs the true value

# Mean of Grouped Data
data_result.groupby('Damage_real')[['Damage_pred']].mean().sort_values(by='Damage_real').plot(style=['x'], mec='r', markersize=10)

# Creating Scatter Plots
plt.scatter(data_result["Damage_real"], data_result["Damage_pred"] , color = 'blue')

# add the title to the plot
plt.title('Linear Regression using SVR Model')
# label x axis
plt.xlabel('Actual Damage')
# label y axis
plt.ylabel('Estimated Damage')
# print the plot
plt.legend([ "Mean of estimated Damage", "Estimated Damage"])

#plt.ylim([0, 90])
#plt.xlim([0, 90])

plt.grid()
#plt.savefig('SVR_Estim_6th.pdf', format='pdf', transparent=True, dpi=300, bbox_inches='tight')
plt.show()

# Replaceing value of DataFrame
data_result.loc[data_result.Damage_real==0.0,'Damage_real']='5,92% Damage'
data_result.loc[data_result.Damage_real==1.0,'Damage_real']='2,96% Damage'
data_result.loc[data_result.Damage_real==2.0,'Damage_real']='8,87% Damage'
data_result.loc[data_result.Damage_real==3.0,'Damage_real']='Healthy'

# Separate data into a series of groups by applying criteria by creating a groupby object
groups_damage = data_result.groupby(data_result.Damage_real)
damage0 = groups_damage.get_group('5,92% Damage')
damage1 = groups_damage.get_group('2,96% Damage')
damage2 = groups_damage.get_group('8,87% Damage')
damage3 = groups_damage.get_group('Healthy')

# Plotting the density plot - visualizing the distribution of observations in dataset,
f, (axes1, axes2, axes3, axes4)=plt.subplots(ncols=4,figsize=(20,10), sharey=True)# sharex=True, sharey=True)

d0 = sns.kdeplot(damage0, ax=axes1, color='blue', label='Estimated',linewidth=4)
d1 = sns.kdeplot(damage1, ax=axes2, color='blue', label='Estimated',linewidth=4)
d2 = sns.kdeplot(damage2, ax=axes3, color='blue', label='Estimated',linewidth=4)
d3 = sns.kdeplot(damage3, ax=axes4, color='blue', label='Estimated',linewidth=4)
#a80 = sns.kdeplot(data_result['Torq_prev80'], ax=axes5, color='blue', label='Estimated (80 cNm)',linewidth=4)

ticks10 = max(d0.get_yticks())
d0.vlines(x=np.mean(damage0['Damage_pred']) , ymin=0, ymax=ticks10, color='red', linestyle='--', label='Mean',linewidth=4)
ticks20 = max(d1.get_yticks())
d1.vlines(x=np.mean(damage1['Damage_pred']) , ymin=0, ymax=ticks20, color='red', linestyle='--', label='Mean',linewidth=4)
ticks30 = max(d2.get_yticks())
d2.vlines(x=np.mean(damage2['Damage_pred']) , ymin=0, ymax=ticks30, color='red', linestyle='--', label='Mean',linewidth=4)
ticks60 = max(d3.get_yticks())
d3.vlines(x=np.mean(damage3['Damage_pred']) , ymin=0, ymax=ticks60, color='red', linestyle='--', label='Mean',linewidth=4)
#ticks80 = max(a80.get_yticks())
#a80.vlines(x=np.mean(data_result['Torq_prev80']) , ymin=0, ymax=ticks80, color='red', linestyle='--', label='Mean',linewidth=4)


d0.set_xlabel("Damage", fontsize=20)
d1.set_xlabel("Damage", fontsize=20)
d2.set_xlabel("Damage", fontsize=20)
d3.set_xlabel("Damage", fontsize=20)
#a80.set_xlabel("Damage", fontsize=20)
d0.set_ylabel("Density", fontsize=20)
d1.set_ylabel("Density", fontsize=20)
d2.set_ylabel("Density", fontsize=20)
d3.set_ylabel("Density", fontsize=20)
#a80.set_ylabel("Density", fontsize=20)
d0.legend(fontsize=12, loc='upper right')
d1.legend(fontsize=12, loc='upper right')
d2.legend(fontsize=12, loc='upper right')
d3.legend(fontsize=12, loc='upper right')
#a80.legend(fontsize=12, loc='upper right')

#a10.set_ylim(0, 0.19)
#a20.set_ylim(0, 0.19)
#a30.set_ylim(0, 0.19)
#a60.set_ylim(0, 0.19)
#a80.set_ylim(0, 0.19)

plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

plt.tight_layout()
#plt.savefig('dist_damage.pdf', format='pdf', transparent=True, dpi=300)