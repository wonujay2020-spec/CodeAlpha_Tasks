#import the Iris dataset
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#Load the dataset
iris = load_iris()

#print basic information
print("Dataset Loaded Successfully!")
print()

print("Feature Names:")

print(iris.feature_names)

print()

print("Target Names:")

print(iris.target_names)

print()

print("Number of Samples:")

print(len(iris.data))

print()

print("First Flower:")

print(iris.data[0])

print()

print("Its Species:")

print(iris.target_names[iris.target[0]])

# Create a DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add the species column
df["species"] = iris.target_names[iris.target]

# Display the first five rows
print(df.head())
print("\nDataset Shape:")
print(df.shape)

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Count each flower species
print("\nFlower Species Count:")
print(df["species"].value_counts())

# ----------------------------
# Data Visualization
# ----------------------------

# Create a scatter plot showing the relationship
#between sepal and petal length
sns.scatterplot(
    data=df,
    x="sepal length (cm)",
    y="petal length (cm)",
    hue="species"
)
plt.title("Sepal Length vs Petal Length")
plt.show()

# Pair Plot
sns.pairplot(df, hue="species")
plt.show()

# Split the data into features (X) and target (y)

X = df.drop("species", axis=1)
y = df["species"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

print("Model Trained Successfully!")

predictions = model.predict(X_test)

print("\nPredictions:")
print(predictions)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
print("\nConfusion Matrix:")
print (cm)

disp.plot()

plt.show()

print("\nProject Completed Successfully!")
