import pandas as pd
import numpy as np

# read the .xlsx file with read_excel function
DM_data = pd.read_excel('DM_Datasets.xlsx')

# update the column names 
DM_data.columns = ['Patient', 'Category', 'Patient location', 'Age', 'Sex', 'Sodium', 'Glucose', 'Blood urea nitrogen',
       'Ethanol by enzymatic assay', 'Measured osmolality', 'OG(no ethanol correction)', 'OG(ethanol correction)',
       'Estimated osmolal contribution', 'Methanol', 'Isopropanol', 'Ethanol', 'Acetone', 'Ethylene glycol', 
       'Propylene glycol', 'Anion Gap', 'Anion > 16', 'Initial level during admission?', 'Clinical History', 'Death',
       'Intravenous ethanol', 'Fomepizole', 'Dialysis', 'Activated charcoal', 'Estimated timing from ingestion to blood draw']
DM_data.head() #display the first 5 rows in data

DM_data.replace("Not performed", np.nan, inplace = True) # Replaces the value "Not performed" with NaN in 'DM_data'.
DM_data.replace("Unknown", np.nan, inplace = True) # Replaces the value "Unknown" with NaN in 'DM_data'.
DM_data.replace("Not available", np.nan, inplace = True) # Replaces the value "Not available" with NaN in 'DM_data'.

# Converts the 'Anion Gap' column in the DataFrame 'DM_data' to numeric values.
# Any non-numeric values will be replaced with NaN.
DM_data['Anion Gap'] = pd.to_numeric(DM_data['Anion Gap'], errors='coerce')

# Fills NaN values in the 'DM_data' with the mean of each numeric column.
# DM_data_ful: Raw data (no preprocessing)
DM_data_ful = DM_data.fillna(DM_data.mean(numeric_only=True)) 
DM_data_ful.head() # Display the first 5 rows the DM_data_ful

import matplotlib.pyplot as plt

# Get unique categories from the 'Category' column
categories = DM_data_ful['Category'].unique()

# Create a subplot layout with 2 rows and 3 columns
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Iterate over the categories and their respective data groups
# Place the data in the subplots, considering the last empty plot
for ax, (name, group) in zip(axes.flatten(), DM_data_ful.groupby('Category')):
    # Scatter plot for each category group
    ax.scatter(group['OG(ethanol correction)'], group['Anion Gap'], label=name)
    ax.set_xlabel("OG(ethanol correction)")
    ax.set_ylabel("Anion Gap")
    ax.legend()
    ax.set_title(name)

# Remove the empty last subplot
for i in range(len(categories), len(axes.flatten())):
    fig.delaxes(axes.flatten()[i])

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

# Import preprocessing module from scikit-learn
from sklearn import preprocessing

# Create the normalizer (MinMaxScaler)
scaler = preprocessing.MinMaxScaler()

# Create a copy of the original data to preserve it
DM_normalization = DM_data_ful.copy()

# Preprocess the features: Age, OG(ethanol correction), Methanol, Isopropanol, Ethanol,
# Acetone, Ethylene glycol, Propylene glycol, and Anion Gap
DM_normalization[['Age','OG(ethanol correction)', 'Methanol','Isopropanol' , 'Ethanol',
                  'Acetone','Ethylene glycol','Propylene glycol','Anion Gap']] = scaler.fit_transform(
                      DM_data_ful[['Age','OG(ethanol correction)', 'Methanol','Isopropanol' ,
                                   'Ethanol','Acetone','Ethylene glycol','Propylene glycol','Anion Gap']])

# Display the first 5 rows the DM_normalization with normalize columns
DM_normalization.head() 

# Create the encoder (OrdinalEncoder)
ordinal_encoder = preprocessing.OrdinalEncoder()

# Preprocess the categorical features: Patient location, Sex, Initial level during admission?,
# Death, Intravenous ethanol, Fomepizole, Dialysis, Activated charcoal
DM_normalization[['Patient location',
                  'Sex',
                  'Initial level during admission?',
                  'Death',
                  'Intravenous ethanol',
                  'Fomepizole',
                  'Dialysis',
                  'Activated charcoal']] = ordinal_encoder.fit_transform(
                      DM_normalization[['Patient location',
                                        'Sex',
                                        'Initial level during admission?',
                                        'Death',
                                        'Intravenous ethanol',
                                        'Fomepizole',
                                        'Dialysis',
                                        'Activated charcoal']])

# Display the first 5 rows the DM_normalization with encode columns
DM_normalization.head() 

# Define the columns to be dropped from the dataframe
drop_columns = ['Patient',
                'Category',
                'Sodium',
                'Glucose',
                'Blood urea nitrogen',
                'Ethanol by enzymatic assay',
                'Measured osmolality',
                'OG(no ethanol correction)',
                'Estimated osmolal contribution',
                'Clinical History',
                'Estimated timing from ingestion to blood draw',
                'Anion > 16']

# Remove the specified columns from the dataframe so it only contains the features needed for our model
dim_reduce_data = DM_normalization.drop(columns=drop_columns)

# Display the first 5 rows the dim_reduce_data
dim_reduce_data.head()

from sklearn.decomposition import PCA

# Create a PCA object with the number of components set to 2
pca = PCA(n_components=2)

# Fit the PCA on the reduced data and transform it to the principal components
DM_data_scaled_pca = pca.fit_transform(dim_reduce_data)

# Convert the principal components into a DataFrame
principal_df = pd.DataFrame(data = DM_data_scaled_pca, columns = ['PC1', 'PC2'])

# Print the shape of the resulting DataFrame to see the dimensions
print(principal_df.shape)

# Display the first few rows of the DataFrame containing the principal components
principal_df.head()

#Import KMeans from scikit-learn
from sklearn.cluster import KMeans

# Create a KMeans object with the number of clusters set to 5, 15 initializations, and 500 max iterations
kmeans = KMeans(n_clusters=5, n_init=15, max_iter=500, random_state=0)

# Train the KMeans model on the reduced data and make cluster predictions
clusters = kmeans.fit_predict(dim_reduce_data)

# Set the figure size for the plot
plt.figure(figsize=(8,6))

# Create a scatter plot of the first two principal components, colored by cluster assignment
plt.scatter(principal_df.iloc[:,0], principal_df.iloc[:,1], c=clusters, cmap="brg", s=40)

# Add title and axis labels to the plot
plt.title('PCA plot in 2D')
plt.xlabel('PC1')
plt.ylabel('PC2')

# Import the t-SNE module from sklearn.manifold
from sklearn.manifold import TSNE

# Create a t-SNE object with n_components=2 to reduce to 2 dimensions
tsne = TSNE(n_components=2)

# Apply the t-SNE algorithm to our dataset and assign the results to X_tsne
# dim_reduce_data is our dataset to be reduced in dimensions
X_tsne = tsne.fit_transform(dim_reduce_data)

# Convert the resulting t-SNE output to a DataFrame
# Name the DataFrame columns 'tsne comp. 1' and 'tsne comp. 2'
tsne_df = pd.DataFrame(data = X_tsne, columns = ['tsne comp. 1', 'tsne comp. 2'])

# Print the shape (number of rows and columns) of the DataFrame
print(tsne_df.shape)

# Display the first 5 rows of the DataFrame
tsne_df.head()

# Create a KMeans object with 5 clusters, 15 initializations, 500 maximum iterations, and a random state for reproducibility
kmeans = KMeans(n_clusters=5, n_init=15, max_iter=500, random_state=0)

# Train the KMeans algorithm on the dataset and predict cluster assignments
clusters = kmeans.fit_predict(dim_reduce_data)

# Set the size of the figure
plt.figure(figsize=(8,6))

# Create a scatter plot of the t-SNE results, colored by the cluster assignments
plt.scatter(tsne_df.iloc[:,0], tsne_df.iloc[:,1], c=clusters, cmap="brg", s=40)

# If you have centroids to plot, you can use the following line (currently commented out)
# plt.scatter(x=centroids_pca[:,0], y=centroids_pca[:,1], marker="x", s=500, linewidths=3, color="black")

# Add titles and labels to the plot for aesthetics
plt.title('t-SNE k-Means')
plt.xlabel('tSNE1')
plt.ylabel('tSNE2')

# Show the plot
plt.show()

# Import the UMAP module from umap
import umap

# Create a UMAP object
um = umap.UMAP()

# Fit the UMAP model to the dataset
X_fit = um.fit(dim_reduce_data)  # We'll use X_fit later

# Transform the dataset using the fitted UMAP model
X_umap = um.transform(dim_reduce_data)

# Convert the UMAP results to a DataFrame
# Name the DataFrame columns 'umap comp. 1' and 'umap comp. 2'
umap_df = pd.DataFrame(data = X_umap, columns = ['umap comp. 1', 'umap comp. 2'])

# Print the shape (number of rows and columns) of the DataFrame
print(umap_df.shape)

# Display the first 5 rows of the DataFrame
umap_df.head()

# Create a KMeans object with 5 clusters, 15 initializations, 500 maximum iterations, and a random state for reproducibility
kmeans = KMeans(n_clusters=5, n_init=15, max_iter=500, random_state=0)

# Train the KMeans algorithm on the dataset and predict cluster assignments
clusters = kmeans.fit_predict(dim_reduce_data)

# Set the size of the figure
plt.figure(figsize=(8,6))

# Create a scatter plot of the UMAP results, colored by the cluster assignments
plt.scatter(umap_df.iloc[:,0], umap_df.iloc[:,1], c=clusters, cmap="brg", s=40)

# Add titles and labels to the plot for aesthetics
plt.title('UMAP k-Means')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')

# Show the plot
plt.show()

# Import the Plotly Express module
import plotly.express as px

# Create a UMAP object with n_components=3 for 3D scatter plot
um = umap.UMAP(n_components=3)

# Fit and transform the dataset using UMAP
components_umap = um.fit_transform(dim_reduce_data)

# Create a 3D scatter plot using Plotly Express
fig = px.scatter_3d(
    components_umap, x=0, y=1, z=2, color=clusters, size=0.1*np.ones(len(dim_reduce_data)), opacity=1,
    title='UMAP plot in 3D',
    labels={'0': 'x', '1': 'y', '2': 'z'},
    width=650, height=500
)

# Display the plot
fig.show()

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import set_link_color_palette

# Assuming umap_df and DM_normalization are already defined and loaded
# Create the hierarchical clustering using the 'ward' linkage method
Z = linkage(umap_df[['umap comp. 1', 'umap comp. 2']], method='ward', metric='euclidean')

# Set a custom color palette
set_link_color_palette(['m', 'c', 'y', 'k', 'b', 'g', 'r'])

# Calculate the color threshold to differentiate more clusters
color_threshold = Z[-5, 2]

# Plot the dendrogram
dendro = dendrogram(
    Z,
    labels=list(DM_normalization['Category']),
    color_threshold=color_threshold,
    above_threshold_color='black'  # Color for the links above the threshold
)

# Set the title and axis labels
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Distance')

# Adjust the font size of x-axis labels
plt.xticks(fontsize=2, rotation=90)

# Save the plot with DPI set to 300
plt.savefig('HC_UMAP.png', dpi=300)

# Show the plot
plt.show()

# set the plot title
plt.title('Dendrogram - 5 clusters')

# set the axis labels
plt.xlabel('Clustering and Sample Counts')
plt.ylabel('distance')

# plot the dendrogram
dendrogram(Z, truncate_mode='lastp', p=5) # no labels possible

plt.savefig('HC_COUNT.png', dpi=300)  # Set DPI to 300 (or any desired value)

# show the plot
plt.show()

from sklearn import preprocessing
#The column of interest was selected as target to create the decision tree.
data_target = DM_data_ful['Patient location']
#With the label encoding function, the expression in the column was converted to 0 and 1.
label_encoder = preprocessing.LabelEncoder()
data_target_encoded = label_encoder.fit_transform(data_target)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42) #create decision tree classifier model
# this model contain information of osmolal and anion gap
dt.fit(DM_normalization[['OG(ethanol correction)',
       'Anion Gap']], data_target_encoded)

import matplotlib.pyplot as plt
from sklearn import tree
# create plot of decision tree
plt.figure(figsize=(200,100))
# tree created with use plot_tree function
tree.plot_tree(dt,
               feature_names=DM_normalization[['OG(ethanol correction)',
       'Anion Gap']].columns.tolist(), 
               class_names=label_encoder.classes_,
              filled = True, 
              rounded=True)
plt.savefig('Decision_Tree.png') 
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

#selected features for random forest model
features = ['OG(ethanol correction)', 'Anion Gap']
X = DM_normalization[features]
y = data_target_encoded

# Create and train the Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Extract feature importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]


plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=np.array(features)[indices])
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

#use cross validation for calculated accuracies
accuracy_data_dt = cross_val_score(dt, X, y, cv=10, scoring='accuracy')
accuracy_data_rf = cross_val_score(rf, X, y, cv=10, scoring='accuracy')

#calculated accuracies for decision tree 
print("Decision Tree Accuracies:")
for i, acc in enumerate(accuracy_data_dt):
    print("Fold {}: Accuracy = {:.2f}%".format(i, acc * 100.0))
print("Average Accuracy for Decision Tree = {:.2f}%".format(accuracy_data_dt.mean() * 100.0))
#calculated accuracies for random forest
print("\nRandom Forest Accuracies:")
for i, acc in enumerate(accuracy_data_rf):
    print("Fold {}: Accuracy = {:.2f}%".format(i, acc * 100.0))
print("Average Accuracy for Random Forest = {:.2f}%".format(accuracy_data_rf.mean() * 100.0))

from sklearn.model_selection import train_test_split
#with train_test_split, train and test data created for data and target data
data_train, data_test, target_train, target_test = train_test_split(
    DM_normalization[['OG(ethanol correction)', 'Anion Gap']], data_target_encoded,test_size=0.2, random_state=42, stratify=data_target)


print("=======TRAIN=========")
display(data_train)
display(target_train)

print("=======TEST=========")
display(data_test)
display(target_test)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Fit the classifier with the training set
dt.fit(data_train, target_train)
 
# Predict on the test set
pred = dt.predict(data_test)
 
# Evaluate the model
accuracy = accuracy_score(target_test, pred)
report = classification_report(target_test, pred, target_names=label_encoder.classes_)
conf_matrix = confusion_matrix(target_test, pred)
 
print(f'Accuracy: {accuracy}')

# True Positives (TP)
TP = conf_matrix[1, 1]
# True Negatives (TN)
TN = conf_matrix[0, 0]
# False Positives (FP)
FP = conf_matrix[0, 1]
# False Negatives (FN)
FN = conf_matrix[1, 0]
# Accuracy excluding True Negatives
accuracy_excl_tn = (TP + FP + FN) / (TP + TN + FP + FN)
print(f'Accuracy (excluding True Negatives): {accuracy_excl_tn}')

print('Classification Report:')
print(report)
print('Confusion Matrix:')
print(conf_matrix)
 
# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

from orangecontrib.associate.fpgrowth import frequent_itemsets
 
# calculate the frequent itemsets

itemsets = dict(frequent_itemsets(DM_normalization[['Intravenous ethanol',
                                            'Fomepizole',
                                            'Dialysis',
                                            'Activated charcoal']].values, 0.01))
 
# store the results in a dataframe
rows = []

for itemset, support_count in itemsets.items():

    domain_names= ",".join([DM_normalization[[ 'Intravenous ethanol',
                                      'Fomepizole',
                                      'Dialysis',
                                      'Activated charcoal']].columns[item_index] for item_index in itemset])

    rows.append((len(itemset), support_count, support_count / len(DM_normalization.index), domain_names))
 
item_set_table = pd.DataFrame(rows, columns=["size", "support count", "support", "items"])

item_set_table.sort_values('support', ascending = False)

item_set_table

from orangecontrib.associate.fpgrowth import association_rules, rules_stats
 
# calculate association rules from the itemsets
rules = association_rules(itemsets, 0.70)
 
# calculate statistics about the rules and store them in a dataframe
rows = []
for premise, conclusion, sup, conf,cov, strength, lift, leverage  in rules_stats(rules, itemsets, len(DM_normalization)):
    premise_names = ",".join([DM_normalization[[ 'Intravenous ethanol',
                                      'Fomepizole',
                                      'Dialysis',
                                      'Activated charcoal']].columns[item_index] for item_index in premise])
    conclusion_names = ",".join([DM_normalization[[ 'Intravenous ethanol',
                                      'Fomepizole',
                                      'Dialysis',
                                      'Activated charcoal']].columns[item_index] for item_index in conclusion])
    rows.append((premise_names, conclusion_names, sup, conf,cov, strength, lift, leverage))
 
pd.DataFrame(rows, columns = ['Premise', 'Conclusion', 'Support', 'Confidence', 'Coverage', 'Strength', 'Lift', 'Leverage'])

# created plots for each group that selected
sns.pairplot(DM_data_ful[['Age',
    'OG(no ethanol correction)',
    'OG(ethanol correction)', 
    'Estimated osmolal contribution',
    'Anion Gap']])

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Features list
features = [
    'Age',
    'OG(no ethanol correction)',
    'OG(ethanol correction)', 
    'Estimated osmolal contribution',
    'Anion Gap'
]

# Separate target variable and arguments
target_variable = 'OG(ethanol correction)'
feature = 'OG(no ethanol correction)'

target = DM_data_ful[target_variable]

# Reshape the argument
features = DM_data_ful[feature].values.reshape(-1, 1)

# Separation into training and testing data sets
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.4, random_state=42
)

# Create and train linear regression model
model = LinearRegression()
model.fit(features_train, target_train)


# Predicted values
target_pred_test = model.predict(features_test)

# Plotting original values
plt.scatter(features_train, target_train, c='green', label='Train')
plt.scatter(features_test, target_test, c='blue', label='Test')

# Plotting predicted values
plt.plot(DM_data_ful[feature], model.predict(DM_data_ful[feature].values.reshape(-1, 1)), c='red', label='Prediction')


plt.xlabel('OG(no ethanol correction)')
plt.ylabel('OG(ethanol correction)')
plt.legend()
plt.show()

# Print the model's formula
print("OG(ethanol correction) = {:.2f} * OG(no ethanol correction) + {:.2f}".format(model.coef_[0], model.intercept_))

# Calculate MSE, RMSE and R^2 values
mse = mean_squared_error(target_test, target_pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(target_test, target_pred_test)

print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R^2: {r2:.2f}")


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

target_variable = 'OG(ethanol correction)'
feature = 'OG(no ethanol correction)'

target = DM_data_ful[target_variable]
features = DM_data_ful[feature].values.reshape(-1, 1)

features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.4, random_state=42
)
# Create and train PolynomialFeatures and LinearRegression model with Pipeline
degree = 2  # Degree of polynomial
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
    ('linear', LinearRegression())
])
pipeline.fit(features_train, target_train)

features_range = np.linspace(features.min(), features.max(), 100).reshape(-1, 1)
prediction_train = pipeline.predict(features_train)
prediction_test = pipeline.predict(features_test)

plt.scatter(features_train, target_train, c='green', label='Train')
plt.scatter(features_test, target_test, c='blue', label='Test')

plt.plot(features_range, pipeline.predict(features_range), c='red', label='Prediction')
plt.xlabel('OG(no ethanol correction)')
plt.ylabel('OG(ethanol correction')
plt.legend()
plt.show()

# Print the model's formula
print(f"{target_variable} = ", end='')
for i, f in enumerate(pipeline.named_steps['poly'].get_feature_names_out([feature])):
    if i > 0:
        print(" + ", end='')
    print(f"{pipeline.named_steps['linear'].coef_[i]}*{f}", end='')
print(f" + {pipeline.named_steps['linear'].intercept_}")


mse = mean_squared_error(target_test, prediction_test)
rmse = np.sqrt(mse)
r2 = r2_score(target_test, prediction_test)

print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R^2: {r2:.2f}")


target_variable = 'Estimated osmolal contribution'
feature = 'OG(no ethanol correction)'
target = DM_data_ful[target_variable]

features = DM_data_ful[feature].values.reshape(-1, 1)

features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.4, random_state=42
)

model = LinearRegression()
model.fit(features_train, target_train)

target_pred_train = model.predict(features_train)
target_pred_test = model.predict(features_test)

plt.scatter(features_train, target_train, c='green', label='Train')
plt.scatter(features_test, target_test, c='blue', label='Test')

plt.plot(DM_data_ful[feature], model.predict(DM_data_ful[feature].values.reshape(-1, 1)), c='red', label='Prediction')

plt.xlabel('OG(no ethanol correction)')
plt.ylabel('Estimated osmolal contribution')
plt.legend()
plt.show()

print("Estimated osmolal contribution = {:.2f} * OG(no ethanol correction) + {:.2f}".format(model.coef_[0], model.intercept_))

mse = mean_squared_error(target_test, target_pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(target_test, target_pred_test)

print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R^2: {r2:.2f}")

target_variable = 'Estimated osmolal contribution'
feature = 'OG(no ethanol correction)'
target = DM_data_ful[target_variable]

features = DM_data_ful[feature].values.reshape(-1, 1)

features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.4, random_state=42
)

degree = 2  
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
    ('linear', LinearRegression())
])
pipeline.fit(features_train, target_train)

features_range = np.linspace(features.min(), features.max(), 100).reshape(-1, 1)
prediction_train = pipeline.predict(features_train)
prediction_test = pipeline.predict(features_test)

plt.scatter(features_train, target_train, c='green', label='Train')
plt.scatter(features_test, target_test, c='blue', label='Test')

plt.plot(features_range, pipeline.predict(features_range), c='red', label='Prediction')
plt.xlabel('OG(no ethanol correction)')
plt.ylabel('Estimated osmolal contribution')
plt.legend()
plt.show()

print(f"{target_variable} = ", end='')
for i, f in enumerate(pipeline.named_steps['poly'].get_feature_names_out([feature])):
    if i > 0:
        print(" + ", end='')
    print(f"{pipeline.named_steps['linear'].coef_[i]}*{f}", end='')
print(f" + {pipeline.named_steps['linear'].intercept_}")


mse = mean_squared_error(target_test, prediction_test)
rmse = np.sqrt(mse)
r2 = r2_score(target_test, prediction_test)

print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R^2: {r2:.2f}")


target_variable = 'Estimated osmolal contribution'
feature = 'OG(ethanol correction)'

target = DM_data_ful[target_variable]

features = DM_data_ful[feature].values.reshape(-1, 1)

features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.4, random_state=42
)

model = LinearRegression()
model.fit(features_train, target_train)

target_pred_train = model.predict(features_train)
target_pred_test = model.predict(features_test)

plt.scatter(features_train, target_train, c='green', label='Train')
plt.scatter(features_test, target_test, c='blue', label='Test')

plt.plot(DM_data_ful[feature], model.predict(DM_data_ful[feature].values.reshape(-1, 1)), c='red', label='Prediction')

plt.xlabel('OG(ethanol correction)')
plt.ylabel('Estimated osmolal contribution')
plt.legend()
plt.show()

print("Estimated osmolal contribution = {:.2f} * OG(ethanol correction)) + {:.2f}".format(model.coef_[0], model.intercept_))

mse = mean_squared_error(target_test, target_pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(target_test, target_pred_test)

print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R^2: {r2:.2f}")


target_variable = 'Estimated osmolal contribution'
feature = 'OG(ethanol correction)'

target = DM_data_ful[target_variable]

features = DM_data_ful[feature].values.reshape(-1, 1)


features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.4, random_state=42
)

degree = 2 
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
    ('linear', LinearRegression())
])
pipeline.fit(features_train, target_train)

features_range = np.linspace(features.min(), features.max(), 100).reshape(-1, 1)
prediction_train = pipeline.predict(features_train)
prediction_test = pipeline.predict(features_test)

plt.scatter(features_train, target_train, c='green', label='Train')
plt.scatter(features_test, target_test, c='blue', label='Test')

plt.plot(features_range, pipeline.predict(features_range), c='red', label='Prediction')
plt.xlabel('OG(ethanol correction)')
plt.ylabel('Estimated osmolal contribution')
plt.legend()
plt.show()

print(f"{target_variable} = ", end='')
for i, f in enumerate(pipeline.named_steps['poly'].get_feature_names_out([feature])):
    if i > 0:
        print(" + ", end='')
    print(f"{pipeline.named_steps['linear'].coef_[i]}*{f}", end='')
print(f" + {pipeline.named_steps['linear'].intercept_}")

target_pred_train = model.predict(features_train)
target_pred_test = model.predict(features_test)

mse = mean_squared_error(target_test, prediction_test)
rmse = np.sqrt(mse)
r2 = r2_score(target_test, prediction_test)

print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R^2: {r2:.2f}")