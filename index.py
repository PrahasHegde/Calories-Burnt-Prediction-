import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error as mae, r2_score as r2


df = pd.read_csv("caloriesburnt.csv")

# print(df.head())

# print(df.shape) # (15000, 9)

# df.isnull().sum()


# print(df.info())

# print(df.describe())

# print(df['Gender'].value_counts()) #number of males and females


#Bargraph for Gender
gender_counts = df['Gender'].value_counts()

# Create the bar chart
plt.figure(figsize=(8, 5))  # Set the figure size (width=5 inches, height=3 inches)
gender_counts.plot(kind='bar')

# Add titles and labels
plt.title('Gender Counts')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()



#Histogram for AGE
age_counts = df['Age'].value_counts()

# Create the bar chart
plt.figure(figsize=(8, 5))  # Set the figure size (width=5 inches, height=3 inches)
age_counts.plot(kind='hist')

# Add titles and labels
plt.title('Age Counts')
plt.xlabel('value')
plt.ylabel('Count')

# Show the plot
plt.show()


#Histogram for Height
height_counts = df['Height'].value_counts()

# Create the bar chart
plt.figure(figsize=(8, 5))  # Set the figure size (width=5 inches, height=3 inches)
height_counts.plot(kind='hist')

# Add titles and labels
plt.title('Height')
plt.xlabel('value')
plt.ylabel('Count')

# Show the plot
plt.show()


#Histogram for Weight
weight_counts = df['Weight'].value_counts()

# Create the bar chart
plt.figure(figsize=(8, 5))  # Set the figure size (width=5 inches, height=3 inches)
weight_counts.plot(kind='hist')

# Add titles and labels
plt.title('Weight')
plt.xlabel('value')
plt.ylabel('Count')

# Show the plot
plt.show()


#Histogram for Body_Temp
Temp_counts = df['Body_Temp'].value_counts()

# Create the bar chart
plt.figure(figsize=(8, 5))  # Set the figure size (width=5 inches, height=3 inches)
Temp_counts.plot(kind='hist')

# Add titles and labels
plt.title('Body_Temp')
plt.xlabel('value')
plt.ylabel('Count')

# Show the plot
plt.show()


#Histogram for Duration
duration_counts = df['Duration'].value_counts()

# Create the bar chart
plt.figure(figsize=(8, 5))  # Set the figure size (width=5 inches, height=3 inches)
duration_counts.plot(kind='hist')

# Add titles and labels
plt.title('Duration')
plt.xlabel('value')
plt.ylabel('Count')

# Show the plot
plt.show()


#Histogram for Heart_Rate
Hrate_counts = df['Heart_Rate'].value_counts()

# Create the bar chart
plt.figure(figsize=(8, 5))  # Set the figure size (width=5 inches, height=3 inches)
Hrate_counts.plot(kind='hist')

# Add titles and labels
plt.title('Heart_Rate')
plt.xlabel('value')
plt.ylabel('Count')

# Show the plot
plt.show()


#neglect the Gender and User_ID column
print(df.iloc[:, 2:])


#correlation Heatmap
plt.figure(figsize=(8, 6))  # Set the figure size
sns.heatmap(df.iloc[:, 2:].corr(), annot=True, cmap='coolwarm', cbar=True)

# Add titles and labels
plt.title('Heatmap')
plt.xlabel('Columns')
plt.ylabel('Rows')

# Show the plot
plt.show()


#Handling Categorical Variable
df.replace({'Gender':{'male':0, 'female':1}}, inplace=True)

#splitting features and labels
y = df['Calories']
X = df.drop(columns=['Calories', 'User_ID'])


#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=123)

X_train.shape, X_test.shape # ((12000, 7), (3000, 7))

y_train.shape, y_test.shape # ((12000,), (3000,))



#Building the Model, Evaluating the Model, and Making Predictions

xgbr = XGBRegressor()
xgbr.fit(X_train, y_train)
xgbr_prediction = xgbr.predict(X_test)


#metrics for evaluation
xgbr_mae = mae(y_test, xgbr_prediction)
print(xgbr_mae) # 1.4849313759878278 

xgbr_r2 = r2(y_test, xgbr_prediction)
print(xgbr_r2) # 0.9988308899957399 


#graph which gives us the difference between the Actual and the Predicted value
plt.figure(figsize=(8, 6))
plt.scatter(y_test, xgbr_prediction, color='blue', alpha=0.5)

# Add titles and labels
plt.title('Actual Value Vs Predicted Value')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')

# Optionally, add a line for perfect prediction
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')

# Show the plot
plt.show()


