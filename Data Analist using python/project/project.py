import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


file_name="housing.csv"
df = pd.read_csv(file_name)
df.head()
print(df.dtypes)#Answer 1
#df.describe()
#print(df.columns)
########################## Data Wrangling ##########################################
#Q2: Dropping 'id' and 'Unnamed: 0' columns and updating df in-place
df.drop(columns=["id"], axis=1, inplace=True)

# Generating statistical summary
summary = df.describe()
print(summary)


############# Replacing####################
#mean=df['bathrooms'].mean()
#df['bathrooms'].replace(np.nan,mean, inplace=True)
#print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
#print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


# Q3: Count 
floor_counts = df['floors'].value_counts().to_frame()

# Rename the column for better readability
floor_counts.columns = ['Number of Houses']

# Display the DataFrame
print(floor_counts)

#Q4:Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='waterfront', y='price', data=df)

# Set titles and labels
plt.title('Price Distribution of Houses with and without Waterfront View')
plt.xlabel('Waterfront View (0 = No, 1 = Yes)')
plt.ylabel('House Price')

# Display the plot
plt.show()

#5 Regression Plot
# Create a regression plot with 'sqft_above' as the independent variable and 'price' as the dependent variable
plt.figure(figsize=(10, 6))
sns.regplot(x='sqft_above', y='price', data=df)

# Set titles and labels
plt.title('Relationship between Square Feet Above and House Price')
plt.xlabel('Square Feet Above')
plt.ylabel('House Price')

# Display the plot
plt.show()

########################################## Model Development #############################


#Q6
# Select the feature and target variable
X = df[['sqft_living']]  # Predictor (independent variable)
y = df['price']           # Target (dependent variable)

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Predict prices using the model
y_pred = model.predict(X)

# Calculate R^2
r2 = r2_score(y, y_pred)
print(f"R^2: {r2}")

#Q7: Fit the features
# Define features and target
features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement", 
            "view", "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]
X = df[features]  # Predictor variables
y = df['price']   # Target variable
# Create and fit the model
model = LinearRegression()
model.fit(X, y)
# Predict prices
y_pred = model.predict(X)
# Calculate R^2
r2 = r2_score(y, y_pred)
print(f"R^2: {r2}")

#Q8: Scaling
# Creating the list of tuples with estimator names and their model constructors
estimators = [
    ('scale', StandardScaler()),
    ('polynomial', PolynomialFeatures(include_bias=False)),
    ('model', LinearRegression())
]

# Display the list
print(estimators)

Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
#Q8
# Create the pipeline
pipeline = Pipeline(estimators)
# Fit the pipeline
pipeline.fit(X, y)
# Calculate R^2
r2 = pipeline.score(X, y)
print(f"R^2: {r2}")
print("done")
#breakpoint()
#######Split Data############
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

#Q9 Create and fit a Ridge regression object using the training data
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and fit the Ridge regression model
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train, y_train)

# Predict using the test data
y_pred = ridge_model.predict(X_test)

# Calculate R^2 on the test data
r2 = r2_score(y_test, y_pred)
print(f"R^2 on test data: {r2}")

# Q10 Polynomial Transformation
#second-order polynomial transform on both the training data and testing data
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Perform a second-order polynomial transform
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
# Create and fit the Ridge regression model
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train_poly, y_train)
# Predict using the test data
y_pred = ridge_model.predict(X_test_poly)
# Calculate R^2 on the test data
r2 = r2_score(y_test, y_pred)
print(f"R^2 on test data: {r2}")
#breakpoint()