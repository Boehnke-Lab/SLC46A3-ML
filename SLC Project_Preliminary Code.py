# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 13:03:30 2024

@author: Amartya Viravalli
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 11:24:59 2024

@author: Amartya Viravalli
"""


import re
import pandas as pd

# Specify the path to your Excel file
excel_file_path = 'C:/Project/input_data.xlsx'

# Specify the sheet name
sheet_name = 'Sheet2'

# Read the Excel sheet into a DataFrame
df_excel = pd.read_excel(excel_file_path, sheet_name)

# Replace NaN values in the 'InputString' column with zero and convert to strings
df_excel['Formulation DETAILS'] = df_excel['Formulation DETAILS'].fillna(0).astype(str)

# Define the regular expression pattern
pattern = r'(\d+(\.\d+)?)\s*([A-Za-z]+)'

# Create an empty list to store results
result_list = []

# Process each row in the DataFrame
for index, row in df_excel.iterrows():
    input_string = row['Formulation DETAILS']
    matches = re.findall(pattern, input_string)
    data_dict = {name: float(number)/100 for number, _, name in matches}
    result_list.append(data_dict)



# Create a DataFrame from the list of dictionaries
df_result = pd.DataFrame(result_list)

# Replace NaN values with zeros in the entire DataFrame
df_result.fillna(0, inplace=True)

# Remove rows containing all zeros
#df_result = df_result.loc[(df_result != 0).any(axis=1)]

# ID = []
# ID.append(df_excel['ID'])

# Zeta = []
# Zeta.append(df_excel['Zeta Potential (mV)'])

# PDI = []
# PDI.append(df_excel['PDI'])

# Size = []
# Size.append(df_excel['Z-Avg (nm)'])


# df_result.insert(0, 'ID', ID[0].astype(str))


# df_result.insert(len(df_result.columns), 'Zeta Potential (mV)', Zeta[0])

# df_result.insert(len(df_result.columns), 'PDI', PDI[0])

# df_result.insert(len(df_result.columns), 'Z-Avg (nm)', Size[0])

for i in range(1,len(df_excel.columns)):
    df_result.insert(len(df_result.columns), df_excel.columns[i], df_excel.iloc[:,i] )

# Print the final DataFrame
print(df_result)

'''
df_result_ones = df_result

df_result_ones = df_result_ones.applymap(lambda x: 1 if x != 0 else x)

# Print the final DataFrame
print(df_result_ones)'''


pc = []
pe = []
pg = []
for i in range(0, len(df_result.columns)):
    if df_result.columns[i][len(df_result.columns[i])-2:len(df_result.columns[i])] == 'PC':
        pc.append(i)
    if df_result.columns[i][len(df_result.columns[i])-2:len(df_result.columns[i])] == 'PE':
        pe.append(i)
    if df_result.columns[i][len(df_result.columns[i])-2:len(df_result.columns[i])] == 'PG':
        pg.append(i)

df_result['PC'] = df_result[df_result.columns[pc[0]]].add(df_result[df_result.columns[pc[1]]])
df_result['PE'] = df_result[df_result.columns[pe[0]]].add(df_result[df_result.columns[pe[1]]])
df_result['PG'] = df_result[df_result.columns[pg[0]]].add(df_result[df_result.columns[pg[1]]])

ds = []
po = []
for i in range(0, len(df_result.columns)):
    if df_result.columns[i][0:2] == 'DS':
        ds.append(i)
    if df_result.columns[i][0:2] == 'PO':
        po.append(i)

ds_to_add = []

for i in range (0,len(ds)):
    ds_to_add.append(df_result.columns[ds[i]])

po_to_add = []

for i in range (0,len(po)):
    po_to_add.append(df_result.columns[po[i]])

df_result['DS'] = df_result[ds_to_add].sum(axis=1)
df_result['PO'] = df_result[po_to_add].sum(axis=1)

print(df_result)

df_result_features = df_result.iloc[0:12, 0:11].copy()
# df_result_features_1 = df_result.iloc[0:12, 7:11].copy()
# df_result_features_2 = df_result.iloc[0:12, 21:25].copy()

# df_result_features = pd.concat([df_result_features_1, df_result_features_2], axis=1)

# Assuming df is your DataFrame
columns_to_delete = ['DSPE', 'POPE']

# Drop specified columns using drop()
df_result_features = df_result_features.drop(columns=columns_to_delete)

# Reset the index
df_result_features = df_result_features.reset_index(drop=True)

replication_factor = 9

# Create a list of DataFrames containing the entire DataFrame
replicated_dfs = [df_result_features.copy() for _ in range(replication_factor)]

# Concatenate the list of DataFrames along the rows axis
result_df = pd.concat(replicated_dfs, ignore_index=True)

df_result_out = df_result.iloc[0:12, 11:20].copy()

# Assuming 'df' is your 12 x 9 DataFrame
df_result_output = pd.DataFrame(df_result_out.T.values.flatten(), columns=['PPC'])

#%%  With Normalization

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from skorch import NeuralNetRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess the data (replace 'your_dataset.csv' with the actual path)
#df = pd.read_csv('your_dataset.csv')

# Separate features and labels
#X = df_result[['PC', 'PE', 'PG']]
#X = df_result[['DSPC', 'DSPE', 'DSPG','POPC', 'POPE', 'POPG', 'Cholesterol']]
X = result_df
y = df_result_output['PPC'] 

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.Tensor(X_train_scaled)
y_train_tensor = torch.Tensor(y_train.values).view(-1, 1)  # Reshape to column vector

X_test_tensor = torch.Tensor(X_test_scaled)
y_test_tensor = torch.Tensor(y_test.values).view(-1, 1)  # Reshape to column vector

# Define a simpler neural network for regression with dropout
class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)  # Dropout layer for regularization
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)  # Dropout layer for regularization
        self.output_layer = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)
        x = self.output_layer(x)
        return x

# Wrap the PyTorch model using skorch
net = NeuralNetRegressor(
    RegressionModel,
    module__input_size=len(X_train.columns),  # Pass the input_size as an argument
    criterion=nn.MSELoss,
    optimizer=optim.Adam,
    lr=0.003,
    max_epochs=100,  # Increased number of epochs for small datasets
    batch_size=len(X_train),  # Set batch size equal to the number of data points
)

# Train the model
train_losses = []  # Track training loss
val_losses = []  # Track validation loss

for epoch in range(net.max_epochs):
    # Train the model
    net.partial_fit(X_train_tensor.numpy(), y_train_tensor.numpy())

    # Evaluate on training set
    with torch.no_grad():
        train_predictions = net.predict(X_train_tensor.numpy())
        train_loss = mean_squared_error(y_train_tensor.numpy(), train_predictions)
        train_losses.append(train_loss)

    # Evaluate on validation set
    with torch.no_grad():
        val_predictions = net.predict(X_test_tensor.numpy())
        val_loss = mean_squared_error(y_test_tensor.numpy(), val_predictions)
        val_losses.append(val_loss)

    # Early stopping: Stop training if validation loss stops improving
    if epoch > 0 and val_losses[-1] > val_losses[-2]:
        print(f'Early stopping at epoch {epoch}')
        break

# Plot learning curves
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

# Evaluate the model
with torch.no_grad():
    test_predictions = net.predict(X_test_tensor.numpy())
    mse = mean_squared_error(y_test_tensor.numpy(), test_predictions)
    mse = mse**0.5

print(f'Root Mean Squared Error on Test Set: {mse}')

# Permutation Importance
perm_importance = permutation_importance(net, X_test_tensor.numpy(), y_test_tensor.numpy())

# Get feature importances and indices
feature_importances = perm_importance.importances_mean
feature_indices = np.argsort(feature_importances)[::-1]

# Display feature importance
print("\nFeature Importance:")
for i, idx in enumerate(feature_indices):
    print(f"Feature {idx + 1}: {feature_importances[idx]}")
    


#%%

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error

# Load and preprocess the data (replace 'your_dataset.csv' with the actual path)
#df = pd.read_csv('your_dataset.csv')

# Separate features and labels
#X = df_result[['DSPC', 'DSPE', 'DSPG','POPC', 'POPE', 'POPG', 'Cholesterol']]
X = result_df
y = df_result_output['PPC']    # Replace 'numeric_label' with the actual name of your numerical label column

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build and train a Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=1000, random_state=42)
rf_regressor.fit(X_train_scaled, y_train)

# Evaluate the model
test_predictions = rf_regressor.predict(X_test_scaled)
mse = mean_squared_error(y_test, test_predictions)

print(f'Mean Squared Error on Test Set: {mse}')

# Permutation Importance
perm_importance = permutation_importance(rf_regressor, X_test_scaled, y_test)

# Get feature importances and indices
feature_importances = perm_importance.importances_mean
feature_indices = np.argsort(feature_importances)[::-1]

# Display feature importance
print("\nFeature Importance:")
for i, idx in enumerate(feature_indices):
    print(f"{X.columns[idx]}: {feature_importances[idx]}")