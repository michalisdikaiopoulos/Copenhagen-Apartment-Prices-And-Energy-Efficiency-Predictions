#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.linalg import svd
from matplotlib.pyplot import figure, legend, plot, title, xlabel, ylabel

# In[ ]:


df = pd.read_csv('bolig_data1.csv')


# In[ ]:


#we start inspecting how our dataset looks like.
df.info()


# In[ ]:


pd.set_option('display.max_rows',500)


# In[ ]:


df.head()


# In[ ]:


pcts = df.isnull().sum()/len(df)*100


# In[ ]:


for null_col,pct in zip(df.columns[pcts>0],pcts[df.columns[pcts>0]]):
    print(f'{null_col}: {pct:.2f}% null')


# In[ ]:


print(df.columns)


# In[ ]:


#'monthly_rent' and 'Månedlig leje' both refer to "monthly rent" (one in English and one in Danish).
#'available_from' and 'Ledig fra' both refer to "available from" (one in English and one in Danish).
# 'Indflytningspris' refer to move_in_price which is the same as the english move_in_price column
# 'Lejeperiode' and rental_period both refer to rental period
# 'Aconto' and monthly_aconto both refer to aconto
# 'move_in_price' is a sum of other variables and we will not include it


# Drop the Danish versions if you want to keep the English ones
df.drop(['Månedlig leje', 'Ledig fra', 'Indflytningspris', 'Lejeperiode', 'Aconto','move_in_price'], axis=1, inplace=True)


# In[ ]:


# Now for easier understanding of which columns we need for our project we will translate the columns from Danish to English
# Dictionary for translating column names
translations = {
    'breadcrumb': 'breadcrumb',
    'title': 'title',
    'description': 'description',
    'address': 'address',
    'monthly_rent': 'monthly_rent',
    'monthly_aconto': 'monthly_aconto',
    'move_in_price': 'move_in_price',
    'available_from': 'available_from',
    'rental_period': 'rental_period',
    'Boligtype': 'housing_type',  # Danish: Boligtype
    'Størrelse': 'size_sqm',  # Danish: Størrelse
    'Værelser': 'rooms',  # Danish: Værelser
    'Etage': 'floor',  # Danish: Etage
    'Møbleret': 'furnished',  # Danish: Møbleret
    'Delevenlig': 'roommate_friendly',  # Danish: Delevenlig
    'Husdyr tilladt': 'pets_allowed',  # Danish: Husdyr tilladt
    'Elevator': 'elevator',  # Danish: Elevator
    'Seniorvenlig': 'senior_friendly',  # Danish: Seniorvenlig
    'Kun for studerende': 'students_only',  # Danish: Kun for studerende
    'Altan/terrasse': 'balcony_terrace',  # Danish: Altan/terrasse
    'Parkering': 'parking',  # Danish: Parkering
    'Opvaskemaskine': 'dishwasher',  # Danish: Opvaskemaskine
    'Vaskemaskine': 'washing_machine',  # Danish: Vaskemaskine
    'Ladestander': 'charging_station',  # Danish: Ladestander
    'Tørretumbler': 'dryer',  # Danish: Tørretumbler
    'Lejeperiode': 'rental_period',  # Danish: Lejeperiode
    'Ledig fra': 'available_from',  # Danish: Ledig fra
    'Månedlig leje': 'monthly_rent',  # Danish: Månedlig leje
    'Aconto': 'aconto',  # Danish: Aconto
    'Depositum': 'deposit',  # Danish: Depositum
    'Forudbetalt husleje': 'prepaid_rent',  # Danish: Forudbetalt husleje
    'Indflytningspris': 'move_in_price',  # Danish: Indflytningspris
    'Oprettelsesdato': 'creation_date',  # Danish: Oprettelsesdato
    'Sagsnr.': 'case_number',  # Danish: Sagsnr.
    'energy_mark_src': 'energy_mark_source',
    'Energimærke ': 'energy_label'  # Danish: Energimærke
}

# Apply the translations to rename columns
df.rename(columns=translations, inplace=True)


# In[ ]:


print(df.columns)


# In[ ]:


#Check for Extra Spaces: It's possible that the column names have leading or trailing spaces, which is common when importing data.
# Let's clean the column names by stripping any unnecessary spaces.

# Strip any leading or trailing spaces from column names
df.columns = df.columns.str.strip()


# In[ ]:


df.info()


# In[ ]:


# We will now try to transform some of object data types to numeric ones. Mostly those that refer to prices.
columns_to_transform=['monthly_rent', 'monthly_aconto', 'deposit', 'prepaid_rent']
# Remove ' kr' and '.' for multiple columns
df[columns_to_transform] = df[columns_to_transform].apply(lambda x: x.str.replace('kr', '').str.replace('.', '').str.replace(',', '').str.strip() if x.str else '0')


# In[ ]:


#df['move_in_price'] = df['move_in_price'].apply(lambda x: x.replace('måneder','').replace('Ubegrænset','0').replace('+','').split('-')[0])


# In[ ]:


# column=['move_in_price']

# df[column]= df[column].apply(lambda x: x.str.replace(' kr', '').str.replace('.', '').str.replace(',', ''))


# In[ ]:


df[columns_to_transform] = df[columns_to_transform].apply(pd.to_numeric)


# In[ ]:


# In the move in price, whenever we have a value between 0 and 24 it's months of rent, so we multiply the months by monthly rent to get the accurate move_in_price and replace it with the number of months
# df.loc[(df['move_in_price']<25)&(df['move_in_price']>0),'move_in_price'] = df.loc[(df['move_in_price']<25)&(df['move_in_price']>0),'move_in_price']*df.loc[(df['move_in_price']<25)&(df['move_in_price']>0),'monthly_rent']


# In[ ]:


# assumption: set prepaid rent to 0 when it's NaN

df['prepaid_rent'] = df['prepaid_rent'].fillna('0').astype(float)


# In[ ]:


df[df.select_dtypes(include=['float']).columns] = df.select_dtypes(include=['float']).astype(int)


# In[ ]:


df['energy_mark'] = df['energy_mark_source'].apply(lambda x: x.split('/')[-1].split('_')[0])


# In[ ]:


df['size_sqm'] = df['size_sqm'].apply(lambda x: x.replace('m²','').strip().split('.')[0]).astype(int)


# In[ ]:


df.drop(columns=['energy_mark_source','energy_label','breadcrumb','title','description','rental_period', 'case_number'], inplace=True)


# In[ ]:


# Replacing As soon as possible in available_from column with the creation_date of the correspondent listing
df.loc[df['available_from'].str.contains('Snarest'),'available_from'] = df.loc[df['available_from'].str.contains('Snarest'),'creation_date']


# In[ ]:

# Dictionary to map Danish month names to numbers
danish_months = {
    " januar ": "1.",
    " februar ": "2.",
    " marts ": "3.",
    " april ": "4.",
    " maj ": "5.",
    " juni ": "6.",
    " juli ": "7.",
    " august ": "8.",
    " september ": "9.",
    " oktober ": "10.",
    " november ": "11.",
    " december ": "12."
}

def format_date(date_str):
    # Try to parse the date with the Danish month name
    for month, number in danish_months.items():
        if month in date_str:
            # Replace the month name with the corresponding number
            date_str = date_str.replace(month, number)
            # Parse the date to a datetime object
            #date_obj = datetime.datetime.strptime(date_str, "%d. %m. %Y")
            #return date_obj.strftime("%d.%m.%Y")
            return date_str
    # If it's already in the correct format
    #date_obj = datetime.datetime.strptime(date_str, "%d.%m.%Y")
    return date_str


# In[ ]:


df['available_from'] = pd.to_datetime(df['available_from'].apply(format_date), dayfirst=True)


# In[ ]:


df['creation_date'] = pd.to_datetime(df['creation_date'], dayfirst=True)


# In[ ]:


df['area'] = df['address'].apply(lambda x: x.split('-')[0].split(',')[-1].strip() if '-' in x else x.split(',')[-1].strip())


# In[ ]:


# We want to make floor a numeric var so we have to make assumptions: Stuen (=living room) is ground floor, Kælder (=cellar) is -1, - is translated to 0 as there is no floor
df['floor'] = df['floor'].apply(lambda x: x.replace('Stuen','0').replace('Kælder','-1').replace('-','0').replace('.','')).astype(int)


# In[ ]:


df.drop(columns=['address'],inplace=True)


# In[ ]:


for dtype, columns in df.columns.to_series().groupby(df.dtypes):
    print(f"Type: {dtype}")
    print(f"Columns: {list(columns)}\n")


# In[ ]:


for col in df.columns:
    if df.dtypes[col] == 'O':
        print('###########################')
        print(col)
        print(df[col].unique(),end='\n\n')


# In[ ]:


# create new column availability_in: buckets of <1 month, 1-3 months, 3+ months

df['availability_in'] = df.apply(lambda x: '<1 month' if (x['available_from']-x['creation_date']).days <30 else ('1-3 months' if (x['available_from']-x['creation_date']).days <90 else '3+ months'), axis = 1)


# In[ ]:


scrape_date = pd.to_datetime('17-09-2024',dayfirst=True)
df['days_on_website'] = df['creation_date'].apply(lambda x: (scrape_date-x).days)


# In[ ]:


df.drop(columns=['available_from','creation_date'],inplace=True)


# In[ ]:


# defined as monthly_rent+aconto (in some cases aconto = 0 and we assume it is included in the rent, so to make the analysis more bulletproof
# we create a new variable total_monthly_rent to not drive misleading results)
df['total_monthly_rent'] = df['monthly_rent'] + df['monthly_aconto']


# In[ ]:


"""
we create a new student_affordable column that, based on the total_monthly_rent,
it examines whether a student can afford renting the specific apartment or not
we were based on different reports on average student salary and other living costs
to estimate the threshold for an affordable option to 7500kr
"""
df['student_affordable'] = df['total_monthly_rent'] < 7500.00


# In[ ]:


np.sum(df['student_affordable'])


# In[ ]:


pd.set_option('display.max_columns',100)


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


continuous_vars = ['monthly_rent','monthly_aconto','size_sqm','deposit','prepaid_rent','total_monthly_rent','days_on_website']
df[continuous_vars] = df[continuous_vars].astype(float)


# In[ ]:


for i,type in enumerate(df.dtypes):
    print('- '+df.columns[i]+': '+str(type).replace('object','discrete/nominal').replace('int64','continuous/ordinal').replace('float64','continuous/ratio').replace('bool', 'discrete/nominal'))


# In[ ]:


df.groupby(df['energy_mark']).count().iloc[:,0]/len(df)*100


# In[ ]:


df['months_on_website'] = df['days_on_website'].apply(lambda x: '<1 month' if x<30 else ('1-3 months' if x<90 else ('3-6 months' if x <180 else '6+ months')))


# In[ ]:


df.to_csv('preprocessed_data.csv', index=False, header=True, encoding='utf-8')

# In[ ]:


df = pd.read_csv('preprocessed_data.csv')


# In[ ]:


## Make directory for plots

Path("./plots").mkdir(exist_ok=True)


# In[ ]:


df.info()


# In[ ]:


df['energy_mark'].value_counts()


# In[ ]:


df[df.select_dtypes(include=['object', 'bool']).columns.tolist()].columns


# ### First Attempt of Summary Statistics

# In[ ]:


from scipy import stats
# Calculate Z-scores
z_scores = np.abs(stats.zscore(df['monthly_rent']))

# Set a threshold (commonly 3)
threshold = 3

# Identify outliers
df[z_scores > threshold]


# In[ ]:


z_scores = np.abs(stats.zscore(df['monthly_aconto']))

# Set a threshold (commonly 3)
threshold = 3

# Identify outliers
df[z_scores > threshold]


# In[ ]:


# 1. Quick Summary Statistics
print("Summary Statistics:")
print(df.describe())


# In[ ]:


continuous_vars = df.select_dtypes(include=['number']).columns.tolist()
continuous_ratio_vars = df.select_dtypes(include=['float64']).columns.tolist()


# In[ ]:


# 2. Correlation Matrix for the numerical variables
correlation_matrix = df[continuous_vars].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)


# In[ ]:


# 3. Heatmap of Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.savefig('plots/cor_matrix_heatmap.png')
plt.subplots_adjust(left=0.35, bottom=0.5)
plt.show()


# In[ ]:


len(df[df['monthly_aconto']==0])/len(df)*100


# In[ ]:


np.max(df['monthly_aconto'])


# In[ ]:


# 4. Scatterplot Matrix with Histograms (only continuous ratio variables)
sns.pairplot(df[df.select_dtypes(include=['float64']).columns.tolist()])
plt.title('Scatterplot Matrix')
plt.show()


# In[ ]:


# distribution of total monthly rent

fig = plt.figure(figsize=(8,8))
sns.histplot(df['total_monthly_rent'], kde=True, color='lightcoral', edgecolor='mistyrose')


# add labels and title
plt.xlabel('Total Monthly Rent')
plt.ylabel('Frequency')
plt.title('Distribution of Total Monthly Rent')

plt.savefig("./plots/distribution_total_monthly_rent.png")

plt.show()


# In[ ]:


for var in df.select_dtypes(include=['float64']).columns.tolist():
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df[var])
    plt.title(f'Box Plot of {var}')
    plt.xlabel(var)
    plt.savefig("./plots/{}_boxplot.png".format(var))
    plt.show()


# In[ ]:


discrete_vars = df.select_dtypes(include=['object']).columns.tolist()
continuous_var = 'total_monthly_rent' # We choose only the dependent variable we later want to predict


# Creating bar charts
for discrete_var in discrete_vars:
    plt.figure(figsize=(8, 5))
    sns.barplot(x=discrete_var, y=continuous_var, data=df, ci=None, color='teal')
    plt.title(f'Bar Chart of Avg. {continuous_var} by {discrete_var}')
    plt.xlabel(discrete_var)
    plt.ylabel(continuous_var)
    plt.show()


# In[ ]:


df.head()


# ### Attempt with the log transformed variables

# We were familiar with a technique to have more interpretable results, which is to transform the data with a function, in this case the log works for us, as the data has a long tail and contains outliers, so applying a log transformation to the variables helps normalize the distribution and make the histograms more interpretable.

# In[ ]:


for var in continuous_ratio_vars:
    if var!='size_sqm' and var!='days_on_website':
        df[f'{var}_log'] = np.log1p(df[var])


# In[ ]:


continuous_ratio_log_vars = [col for col in df.select_dtypes(include=['float64']).columns.tolist() if ('log' in col or col=='size_sqm' or col=='days_on_website')]


# In[ ]:


# 1. Quick Summary Statistics
print("Summary Statistics:")
print(df.describe())


# In[ ]:


# 2. Correlation Matrix for the numerical variables
correlation_matrix_vars = continuous_ratio_log_vars.extend(['floor','rooms'])
correlation_matrix = df[continuous_ratio_log_vars].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)


# In[ ]:


# 3. Heatmap of Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.subplots_adjust(left=0.2, bottom=0.2)
plt.savefig('plots/log_cor_matrix_heatmap.png')
plt.show()


# In[ ]:


# 4. Scatterplot Matrix with Histograms (only continuous ratio variables)
sns.pairplot(df[continuous_ratio_log_vars])
plt.title('Scatterplot Matrix')
plt.show()


# In[ ]:


# Logarithmic distribution of total monthly rent

fig = plt.figure(figsize=(8,8))
sns.histplot(df['total_monthly_rent_log'], kde=True, color='lightcoral', edgecolor='mistyrose')


# add labels and title
plt.xlabel('Total Monthly Rent (Log)')
plt.ylabel('Frequency')
plt.title('Distribution of log-transformed Total Monthly Rent')

plt.savefig("./plots/log_distribution_total_monthly_rent.png")

plt.show()


# In[ ]:


for var in continuous_ratio_log_vars:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df[var], color='teal')
    plt.title(f'Box Plot of log-transformed {var}')
    plt.xlabel(var)
    plt.savefig("./plots/{}_log_boxplot.png".format(var))
    plt.show()


# In[ ]:


# 5. Bar charts of continuous variables by discrete variables
discrete_vars = df.select_dtypes(include=['object']).columns.tolist()
continuous_var = 'total_monthly_rent' # We choose only the dependent variable we later want to predict

# Creating bar charts
for discrete_var in discrete_vars:
    plt.figure(figsize=(8, 5))
    sns.barplot(x=discrete_var, y=continuous_var, data=df, color='teal')
    plt.title(f'Bar Chart of Avg. {continuous_var} by {discrete_var}')
    plt.xlabel(discrete_var)
    plt.ylabel(continuous_var)
    plt.show()


# In[ ]:


df1 = df.drop(columns=[col for col in continuous_ratio_vars if (col!='size_sqm' and col!='days_on_website')])


# In[ ]:


df1.to_csv('preprocessed_log_data.csv', index=False, header=True, encoding='utf-8')

# In[ ]:


df = pd.read_csv('preprocessed_data.csv')

# In[ ]:


continuous_vars = df.select_dtypes(include=['float64']).columns.tolist()
continuous_vars

# In[ ]:


pca_variable_subset = df[continuous_vars]

# In[ ]:


pca_variable_subset.info()

# ### PCA

# In[ ]:


X = pca_variable_subset.to_numpy()
N = X.shape[0]

# In[ ]:


mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_centered = (X - mean) / std

# Step 2: Compute covariance matrix
cov_matrix = np.cov(X_centered.T)

# Step 3: Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Step 4: Sort eigenvalues and eigenvectors
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Step 5: Project data onto principal components
X_pca = np.dot(X_centered, eigenvectors[:, :2])

# Results
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
print("Projected Data (PCA):\n", X_pca)

# In[ ]:


rho = eigenvalues / eigenvalues.sum()
threshold = 0.9

for i in range(len(eigenvalues)):
    if np.cumsum(rho)[i] > threshold:
        print(f'We need {i + 1} components to explain at least 90% of the variance of the data')
        break

# Plot variance explained
plt.figure()
plt.plot(range(1, len(rho) + 1), rho, "x-")
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
plt.plot([1, len(rho)], [threshold, threshold], "k--")
plt.title("Variance explained by principal components")
plt.xlabel("Number of principal components")
plt.ylabel("Variance explained")
plt.legend(["Individual", "Cumulative", "Threshold"])
plt.grid()
plt.savefig("./plots/pca_variance_threshold")
plt.show()

# ### SVD

# In[ ]:


# Subtract mean value from data
Y = (X - np.ones((N, 1)) * X.mean(axis=0)) / X.std(axis=0)

# PCA by computing SVD of Y
U, S, V = svd(Y, full_matrices=False)

# Compute variance explained by principal components
rho = (S * S) / (S * S).sum()

threshold = 0.90

for i in range(len(rho)):
    if np.cumsum(rho)[i] > threshold:
        print(f'{i + 1} components/variables needed to surpass the threshold={threshold}')
        break

# Plot variance explained
plt.figure()
plt.plot(range(1, len(rho) + 1), rho, "x-")
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
plt.plot([1, len(rho)], [threshold, threshold], "k--")
plt.title("Variance explained by principal components")
plt.xlabel("Principal component")
plt.ylabel("Variance explained")
plt.legend(["Individual", "Cumulative", "Threshold"])
plt.grid()
plt.show()

# In[ ]:


loadings_pc1 = V[0, :]
loadings_pc2 = V[1, :]

# In[ ]:


# Get the indices of the top 5 highest values
top_indices1 = np.argsort(loadings_pc1)[-5:]  # Get last 5 indices after sorting
top_indices_sorted1 = top_indices1[np.argsort(-loadings_pc1[top_indices1])]

top_indices2 = np.argsort(loadings_pc2)[-5:]  # Get last 5 indices after sorting
top_indices_sorted2 = top_indices2[np.argsort(-loadings_pc2[top_indices2])]

# In[ ]:


print('First Principal Component:\n')

for idx in top_indices_sorted1:
    print(f'{pca_variable_subset.columns[idx]} with coefficient: {loadings_pc1[idx]}', end='\n')

print('\n#############################################################')
print('\nSecond Principal Component:\n')

for idx in top_indices_sorted2:
    print(f'{pca_variable_subset.columns[idx]} with coefficient: {loadings_pc2[idx]}', end='\n')

# In[ ]:


df1 = pd.read_csv('preprocessed_data.csv')

# In[ ]:


df1.info()

# In[ ]:


classLabels = df1['student_affordable'].tolist()
classNames = set(classLabels)
classDict = dict(zip(classNames, range(len(classNames))))

# Extract vector y, convert to NumPy array
y = np.asarray([classDict[value] for value in classLabels])

# In[ ]:


# Project the centered data onto principal component space
Z = Y @ V.T

# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = figure()
title("Copenhagen Apartments/Rooms data: PCA")
# Z = array(Z)
for c in range(len(classNames)):
    # select indices belonging to class c:
    class_mask = y == c
    plot(Z[class_mask, i], Z[class_mask, j], "o", alpha=0.5)
legend(classNames)
xlabel("PC{0}".format(i + 1))
ylabel("PC{0}".format(j + 1))

# In[ ]:


# Project the centered data onto principal component space
Z = Y @ V.T

# Indices of the principal components to be plotted
i = 0
j = 1
k = 2

# Creating figures for the plot
fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection='3d')

for c in range(len(classNames)):
    # select indices belonging to class c:
    class_mask = y == c
    ax.scatter3D(Z[class_mask, i], Z[class_mask, j], Z[class_mask, k], "o", alpha=0.5)

plt.title("Copenhagen Apartments/Rooms data: PCA")
legend(classNames, title="Student affordable")
ax.set_xlabel("PC{0}".format(i + 1))
ax.set_ylabel("PC{0}".format(j + 1))
ax.set_zlabel("PC{0}".format(k + 1))

# Change plot angle
ax.view_init(10, -140)

# Save plot
plt.savefig("./plots/pca_projection.png")

# display the  plot
plt.show()