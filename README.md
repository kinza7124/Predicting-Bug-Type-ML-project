# Complete Code Explanation for Bug Classification Project

---

## **PROJECT OVERVIEW**

This project is about **predicting the type of bug** from a bug tracking system. The dataset contains information about software bugs, and we want to automatically classify each bug into one of three categories:
- **Defect**: A bug or error in the software
- **Task**: A task that needs to be done
- **Enhancement**: A feature improvement request

---

## **PART 1: DATA LOADING AND EXPLORATION**

### **What is happening here?**

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("bugs-2025-02-23.csv")
```

**Explanation:**
- **pandas**: Library for working with data tables (like Excel but for programming)
- **matplotlib/seaborn**: Libraries for creating graphs and visualizations
- **read_csv()**: Reads the CSV file and stores it in a variable called `df` (dataframe)

### **Understanding the Data**

```python
print(df.shape)        # Shows (rows, columns) → (10000, 9)
print(df.info())       # Shows data types and memory usage
print(df.describe())   # Shows statistics like mean, min, max
print(df.head())       # Shows first 5 rows
```

**What each command does:**
- **shape**: Tells us we have 10,000 bug reports with 9 features
- **info()**: Shows what type of data each column contains (text, numbers, dates)
- **describe()**: Shows numerical statistics (only works on number columns)
- **head()**: Displays the first few rows so we can see what the data looks like

**Original Columns:**
1. Bug ID - Unique identifier (not useful for prediction)
2. Type - What we want to predict (defect/task/enhancement)
3. Summary - Text description of the bug
4. Product - Which software product has the bug
5. Component - Which part of the product
6. Assignee - Who is assigned (not useful for prediction)
7. Status - Current status (not useful - comes after prediction)
8. Resolution - How it was resolved (not useful - comes after prediction)
9. Updated - Date when bug was updated

---

## **PART 2: DATA CLEANING**

### **Step 1: Removing Unnecessary Columns**

```python
new_df1 = df.drop(columns=["Bug ID", "Status", "Resolution", "Assignee"])
```

**Why?**
- **Bug ID**: Just a number, doesn't help predict type
- **Status/Resolution**: These are decided AFTER we know the bug type
- **Assignee**: Who is assigned doesn't affect what type of bug it is

**Result**: We keep only useful columns: Type, Summary, Product, Component, Updated

---

### **Step 2: Converting Data Types**

```python
new_df1["Type"] = new_df1["Type"].astype("category")
new_df1["Summary"] = new_df1["Summary"].astype("string")
new_df1["Product"] = new_df1["Product"].astype("category")
new_df1["Component"] = new_df1["Component"].astype("category")
new_df1["Updated"] = pd.to_datetime(new_df1["Updated"], errors="coerce")
```

**Why change data types?**
- **category**: More memory-efficient for columns with repeated values (like Type, Product)
- **string**: Ensures text data is treated as strings
- **to_datetime()**: Converts date strings into actual date objects so we can extract year/month

---

### **Step 3: Feature Engineering - Extracting Year and Month**

```python
new_df1["Year"] = new_df1["Updated"].dt.year
new_df1["Month"] = new_df1["Updated"].dt.month
new_df1 = new_df1.drop(columns=["Updated"])
```

**What is Feature Engineering?**
Creating new useful features from existing data.

**Why extract Year and Month?**
- Different years/months might have different types of bugs
- We can't use the full date directly, but year/month might show patterns
- Example: Maybe more defects in certain months?

**dt.year** and **dt.month**: Extract year and month from the date column

---

### **Step 4: Handling Product Column**

```python
top_products = new_df1["Product"].value_counts().nlargest(15).index
new_df1["Product_grouped"] = new_df1["Product"].apply(
    lambda x: x if x in top_products else "Other"
)
```

**Problem**: Too many different product names (some appear only once or twice)

**Solution**: 
- Find top 15 most common products
- Keep those 15 as they are
- Group all others into "Other" category

**Why?**
- Too many categories make the model complex and slow
- Rare products don't have enough data to learn from
- Grouping similar rare items helps the model generalize

**How it works:**
- `value_counts()`: Counts how many times each product appears
- `nlargest(15)`: Gets the 15 most common
- `lambda x: ...`: Applies a rule to each value (keep if in top 15, else change to "Other")

---

## **PART 3: EXPLORATORY DATA ANALYSIS (EDA)**

### **1. Checking for Missing Values**

```python
new_df1.isna().sum()
```

**What it does:** Counts missing values in each column
**Result:** No missing values found (good!)

---

### **2. Checking Class Balance**

```python
new_df1["Type"].value_counts()
```

**Result:**
- Defect: 6,712 (67%)
- Task: 2,280 (23%)
- Enhancement: 1,008 (10%)

**Problem:** Classes are **imbalanced** (defect has way more examples)

**Why is this bad?**
- Model might learn to always predict "defect" and still get high accuracy
- It won't learn to distinguish between the three types properly

**Solution:** Use SMOTE to balance (explained later)

---

## **PART 4: DATA PREPROCESSING**

### **Why Preprocessing is Needed?**

Computers can't directly understand text like "bug crashes when opening file". We need to convert:
- **Text** → Numbers (TF-IDF)
- **Categories** → Numbers (One-Hot Encoding)
- **Dates** → Numbers (already done - Year and Month)

---

### **Step 1: Separating Features and Target**

```python
X = new_df1.drop("Type", axis=1)  # Features (what we use to predict)
y = new_df1["Type"]                # Target (what we want to predict)
```

- **X**: Input features (Summary, Product, Component, Year, Month)
- **y**: Output we want to predict (Type: defect/task/enhancement)

---

### **Step 2: Creating a Preprocessing Pipeline**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(max_features=500), "Summary"),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)
```

**ColumnTransformer**: Applies different preprocessing to different columns

**Three Transformers:**

1. **TF-IDF Vectorizer (for Summary text)**
   - Converts text into numbers
   - **TF-IDF** = Term Frequency-Inverse Document Frequency
   - Measures how important a word is in a document
   - `max_features=500`: Only keeps top 500 most important words
   - Example: "crash error bug" → [0.5, 0.3, 0.2, 0, 0, ...] (500 numbers)

2. **One-Hot Encoder (for Product and Component)**
   - Converts categories into binary columns
   - Example: Product = "Firefox" → [0, 1, 0, 0, 0]
   - Example: Product = "Core" → [1, 0, 0, 0, 0]
   - `handle_unknown="ignore"`: If new category appears, ignore it (set all to 0)

3. **Passthrough (for Year and Month)**
   - Keeps numeric columns as they are (no transformation)

**Result:** All features converted to numbers that the model can understand

---

### **Step 3: Applying Preprocessing**

```python
X_processed = preprocessor.fit_transform(X)
```

- **fit()**: Learns the transformation rules from training data
- **transform()**: Applies those rules to convert data
- **fit_transform()**: Does both in one step

---

## **PART 5: HANDLING CLASS IMBALANCE WITH SMOTE**

### **What is SMOTE?**

**SMOTE** = Synthetic Minority Oversampling Technique

**Problem:**
- Defect: 6,712 examples
- Task: 2,280 examples  
- Enhancement: 1,008 examples

**Solution:**
SMOTE creates fake (synthetic) examples of minority classes to balance them.

**How it works:**
1. Takes existing examples from minority class
2. Finds nearest neighbors
3. Creates new examples between them
4. Results in balanced classes (all three have similar counts)

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_processed, y)
```

**Result:** All three classes now have approximately 6,712 examples each

---

## **PART 6: VISUALIZATIONS**

### **Visualization 1: Class Distribution (Before/After SMOTE)**

```python
plt.subplot(1,2,1)
sns.countplot(x=y, order=class_order)
plt.title("Class Distribution Before SMOTE")

plt.subplot(1,2,2)
sns.countplot(x=y_resampled, order=class_order)
plt.title("Class Distribution After SMOTE")
```

**What it shows:** Bar chart comparing class counts before and after balancing

---

### **Visualization 2: Bugs per Year**

```python
sns.countplot(x=new_df1["Year"], order=sorted(new_df1["Year"].unique()))
```

**What it shows:** How many bugs were reported each year (trend over time)

---

### **Visualization 3: Bug Type Distribution Across Products**

```python
sns.countplot(data=new_df1, x="Product_grouped", hue="Type")
```

**What it shows:** Stacked bar chart showing which products have more defects/tasks/enhancements

---

### **Visualization 4: WordCloud**

```python
from wordcloud import WordCloud

text = " ".join(new_df1["Summary"].dropna().astype(str))
wordcloud = WordCloud(width=800, height=400).generate(text)
plt.imshow(wordcloud)
```

**What it shows:** Visual representation of most common words in bug summaries (bigger = more frequent)

---

### **Visualization 5: Heatmap (Bugs per Product per Year)**

```python
pivot = new_df1.pivot_table(index="Product_grouped", columns="Year", 
                           values="Summary", aggfunc="count")
sns.heatmap(pivot, annot=True)
```

**What it shows:** 
- Rows = Products
- Columns = Years
- Colors = Number of bugs (darker = more bugs)
- Helps identify spikes in bug reports

---

### **Visualization 6: Top Words per Bug Type**

```python
vectorizer = CountVectorizer(stop_words="english", max_features=20)
for bug_type in new_df1["Type"].unique():
    summaries = new_df1[new_df1["Type"]==bug_type]["Summary"]
    # ... find top words ...
    sns.barplot(x=top_words.values, y=top_words.index)
```

**What it shows:** For each bug type, which words appear most frequently
- Helps understand language differences between defect/task/enhancement

---

## **PART 7: CORRELATION ANALYSIS**

### **What is Correlation?**

Correlation measures how two features are related:
- **+1.0**: Perfect positive relationship (both increase together)
- **0.0**: No relationship
- **-1.0**: Perfect negative relationship (one increases, other decreases)

```python
corr_matrix = df[["Year", "Month", "Type_encoded"]].corr()
sns.heatmap(corr_matrix, annot=True)
```

**What it shows:** Heatmap with correlation values between features
- Example: Year vs Bug Type = 0.37 (moderate positive correlation)
- This means newer years might have different bug type distributions

---

## **PART 8: K-NEAREST NEIGHBORS (KNN) CLASSIFIER**

### **What is KNN?**

**K-Nearest Neighbors**: A simple classification algorithm

**How it works:**
1. When you have a new bug to classify
2. Find the K closest (most similar) bugs in training data
3. Look at what type those K bugs are
4. Predict the most common type among those K neighbors

**Example:** If K=5, find 5 most similar bugs. If 4 are "defect" and 1 is "task", predict "defect"

---

### **Step 1: Data Splitting**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, 
    test_size=0.2,              # 20% for testing
    random_state=42,            # For reproducibility
    stratify=y_resampled        # Maintains class balance in splits
)
```

**Why split?**
- **Training set (80%)**: Used to teach the model
- **Test set (20%)**: Used to evaluate performance (model has never seen this data)

**stratify**: Ensures both sets have same class distribution (important for imbalanced data)

---

### **Step 2: Feature Scaling**

```python
scaler = StandardScaler()
X_resampled[:, -2:] = scaler.fit_transform(X_resampled[:, -2:])
```

**Why scale?**
- Year values (1999-2025) are much larger than Month values (1-12)
- Without scaling, Year would dominate distance calculations
- StandardScaler converts to mean=0, std=1 (normalized)

**Example:**
- Before: Year=2020, Month=6
- After: Year=1.5, Month=0.2 (both on same scale)

---

### **Step 3: Choosing K Value**

```python
initial_k = int(np.sqrt(n_train))  # Rule of thumb: sqrt of training samples
k_values = list(range(max(1, initial_k-10), initial_k+11, 2))
```

**Why test different K values?**
- **K too small** (like K=1): Too sensitive to noise, overfitting
- **K too large** (like K=1000): Too general, underfitting
- **K odd numbers**: Avoids ties in binary classification (not critical for 3 classes)

**Rule of thumb:** Start with √(number of training samples)

**Testing process:**
- Try different K values
- Train model with each K
- Test accuracy
- Choose K with highest accuracy

---

### **Step 4: KNN Parameters**

```python
knn = KNeighborsClassifier(
    n_neighbors=k,              # Number of neighbors to check
    weights='distance',         # Closer neighbors count more
    metric='cosine',            # Distance measure
    n_jobs=-1                   # Use all CPU cores
)
```

**Parameters explained:**

1. **weights='distance'**
   - Closer neighbors have more influence on prediction
   - Example: If closest neighbor is defect, it counts more than far neighbor

2. **metric='cosine'**
   - How to measure "distance" between bugs
   - **Cosine**: Good for high-dimensional data (like TF-IDF vectors)
   - Measures angle between vectors, not absolute distance
   - Better than Euclidean for text data

3. **n_jobs=-1**
   - Uses all available CPU cores for faster computation

---

### **Step 5: Validation Set Approach**

```python
# Split: Train (70%) → Validation (10%) → Test (20%)
X_train_temp, X_test, y_train_temp, y_test = train_test_split(...)
X_train, X_val, y_train, y_val = train_test_split(X_train_temp, ...)
```

**Why three sets?**
- **Training**: Learn the model
- **Validation**: Choose best hyperparameters (like K)
- **Test**: Final evaluation (only touched once, at the end)

**Process:**
1. Train model with different K values
2. Test each on validation set
3. Pick best K (highest validation accuracy)
4. Retrain with best K using training + validation
5. Final test on test set (unbiased estimate)

---

## **PART 9: OTHER CLASSIFICATION MODELS**

### **Why Try Multiple Models?**

Different algorithms work better for different problems. Let's compare:

---

### **1. Decision Tree Classifier**

```python
DecisionTreeClassifier(random_state=42)
```

**How it works:**
- Creates a tree of yes/no questions
- Example: "Does Summary contain 'crash'?" → Yes → "Is Product Firefox?" → Predict defect
- Easy to interpret, but can overfit

**Accuracy: 83.81%**

---

### **2. Random Forest Classifier**

```python
RandomForestClassifier(random_state=42, n_jobs=-1)
```

**How it works:**
- Creates MANY decision trees (ensemble)
- Each tree votes on the prediction
- Final prediction = majority vote
- More robust than single tree

**Accuracy: 91.73%** (Best!)

**Why better?**
- Multiple trees reduce overfitting
- More stable predictions

---

### **3. Gaussian Naive Bayes**

```python
GaussianNB()
```

**How it works:**
- Uses probability and Bayes' theorem
- Assumes features are independent (naive assumption)
- Fast but simple

**Accuracy: 62.51%** (Worst)

**Why worse?**
- Assumption of independence is too strong for this data
- Text features are highly correlated

---

### **4. Logistic Regression**

```python
LogisticRegression(max_iter=500, n_jobs=-1)
```

**How it works:**
- Uses a mathematical formula to find best line/plane separating classes
- Linear model (assumes linear relationships)
- Fast and interpretable

**Accuracy: 80.98%**

---

### **5. Gradient Boosting Classifier**

```python
GradientBoostingClassifier(random_state=42)
```

**How it works:**
- Creates trees sequentially
- Each new tree fixes errors of previous trees
- Powerful but slower

**Accuracy: 78.40%**

---

### **6. SVM (Support Vector Machine)**

```python
SVC(kernel='linear', probability=True, random_state=42)
```

**How it works:**
- Finds the best boundary (hyperplane) separating classes
- Tries to maximize margin between classes
- Good for high-dimensional data

**Accuracy: 81.53%**

---

### **7. Ensemble: Voting Classifier**

```python
VotingClassifier(
    estimators=[('rf', RandomForest), ('gb', GradientBoosting), ...],
    voting='soft'
)
```

**How it works:**
- Combines multiple models
- Each model makes prediction
- Final prediction = majority vote (hard) or weighted average (soft)

**Why ensemble?**
- Different models catch different patterns
- Combining them often improves accuracy
- More robust to errors

---

### **8. Stacking Classifier**

```python
StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner
)
```

**How it works (2-level learning):**

**Level 1 (Base Models):**
- Train multiple models (RF, GB, LR, SVM)
- Each makes predictions

**Level 2 (Meta Model):**
- Takes predictions from Level 1 as input
- Learns which base model to trust for which cases
- Makes final prediction

**Example:**
- Base models predict: [defect, defect, task, defect]
- Meta model learns: "When RF and GB agree, trust them"
- Final: defect

**Accuracy: 94.34%** (Best overall!)

---

## **PART 10: K-MEANS CLUSTERING (UNSUPERVISED LEARNING)**

### **What is Clustering?**

**Supervised Learning** (what we did before):
- We know the correct answers (defect/task/enhancement)
- Model learns from labeled examples

**Unsupervised Learning** (clustering):
- We DON'T know the correct answers
- Model finds patterns and groups similar bugs together

---

### **What is K-Means?**

Groups data into K clusters based on similarity.

**How it works:**
1. Randomly place K cluster centers
2. Assign each bug to nearest cluster
3. Move cluster center to average of its bugs
4. Repeat steps 2-3 until clusters don't change

**Goal:** Bugs in same cluster are similar to each other

---

### **Choosing K (Elbow Method)**

```python
inertia = []
for k in range(2, 11):
    km = KMeans(n_clusters=k)
    km.fit(X_cluster)
    inertia.append(km.inertia_)
```

**Inertia**: Sum of squared distances from bugs to their cluster center
- Lower inertia = tighter clusters (better)

**Elbow Method:**
- Plot inertia vs K
- Look for "elbow" (point where improvement slows)
- Choose K at the elbow

**In this project:** K=3 (matches the 3 bug types)

---

### **Visualizing Clusters with PCA**

```python
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cluster)
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=cluster_labels)
```

**Problem:** We have 4 features, but can only plot 2D

**Solution:** PCA (Principal Component Analysis)
- Reduces dimensions while keeping most information
- Projects 4D data onto 2D plane
- Can visualize clusters

---

### **Comparing Clusters with Actual Bug Types**

```python
comparison = pd.crosstab(cluster_df["Type"], cluster_df["Cluster"])
```

**What it shows:**
- Do clusters match actual bug types?
- If Cluster 0 has mostly defects, clustering found the pattern!
- Creates a confusion matrix: clusters vs actual types

---

## **PART 11: NEURAL NETWORKS**

### **1. Perceptron (Single-Layer Neural Network)**

```python
Perceptron(max_iter=1000, random_state=42)
```

**What is a Perceptron?**
- Simplest neural network
- Single layer of neurons
- Can only learn linear patterns

**Accuracy: 33.33%** (Very poor!)

**Why so bad?**
- Bug classification is complex (non-linear)
- Single layer can't capture complex relationships
- Basically predicting all as "defect" (most common class)

---

### **2. Multi-Layer Perceptron (MLP)**

```python
MLPClassifier(
    hidden_layer_sizes=(64, 32),  # 2 hidden layers with 64 and 32 neurons
    activation="relu",             # Activation function
    max_iter=50                    # Maximum training iterations
)
```

**Architecture:**
- **Input Layer**: Receives features (500 TF-IDF + encoded categories + Year/Month)
- **Hidden Layer 1**: 64 neurons
- **Hidden Layer 2**: 32 neurons
- **Output Layer**: 3 neurons (one for each bug type)

**How it works:**
1. Data flows forward through layers
2. Each neuron applies: `output = activation(weighted_sum + bias)`
3. **ReLU activation**: `max(0, x)` - introduces non-linearity
4. Output layer gives probabilities for each class
5. Backpropagation: Adjusts weights to minimize errors

**Accuracy: 75.62%** (Much better!)

**Why better than Perceptron?**
- Multiple layers can learn complex patterns
- Non-linear activation allows curved decision boundaries
- Can capture relationships between features

---

### **What is Backpropagation?**

**Backpropagation** = Backward propagation of errors

**Process:**
1. Forward pass: Make prediction
2. Calculate error (difference from true label)
3. Backward pass: Propagate error back through layers
4. Adjust weights to reduce error
5. Repeat

**Analogy:** Like adjusting dials on a radio to get clear signal - you adjust weights to get better predictions

---

### **Error Metrics for Classification**

```python
accuracy_score()      # Overall correctness
precision_score()     # Of predicted defects, how many are actually defects?
recall_score()        # Of actual defects, how many did we catch?
f1_score()           # Balance between precision and recall
```

**Example:**
- **Precision (defect)**: Of 100 predicted defects, 90 are actually defects → 90% precision
- **Recall (defect)**: There are 100 actual defects, we found 85 → 85% recall
- **F1 Score**: Harmonic mean of precision and recall

**Why multiple metrics?**
- Accuracy alone can be misleading with imbalanced classes
- Precision/Recall/F1 give more detailed picture

---

## **PART 12: PCA VISUALIZATION**

```python
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_resampled.toarray())
```

**Why use PCA?**
- Data has hundreds of dimensions (500 TF-IDF features + encoded categories)
- Hard to visualize or understand
- PCA reduces to 2-3 dimensions while keeping most important information

**How it works:**
- Finds directions of maximum variance
- Projects data onto these directions
- Most variance = most information

**Result:** 2D scatter plot showing how bugs are distributed
- Similar bugs cluster together
- Different bug types might form separate groups

---

## **KEY CONCEPTS SUMMARY**

### **1. Supervised vs Unsupervised Learning**
- **Supervised**: Learn from labeled examples (Classification: defect/task/enhancement)
- **Unsupervised**: Find patterns without labels (Clustering: group similar bugs)

### **2. Classification vs Clustering**
- **Classification**: Predict category (defect/task/enhancement) - needs labels
- **Clustering**: Group similar items - no labels needed

### **3. Feature Engineering**
- Creating new useful features from existing data
- Example: Extracting Year/Month from date

### **4. Preprocessing**
- Converting data into format models can understand
- Text → TF-IDF vectors
- Categories → One-Hot encoding
- Numbers → Scaling

### **5. Overfitting vs Underfitting**
- **Overfitting**: Model memorizes training data, fails on new data
- **Underfitting**: Model too simple, can't learn patterns

### **6. Train-Validation-Test Split**
- **Train**: Learn model parameters
- **Validation**: Tune hyperparameters (like K)
- **Test**: Final unbiased evaluation

### **7. Ensemble Methods**
- Combine multiple models for better accuracy
- Examples: Voting, Stacking, Random Forest

---

## **WHY THIS PROJECT IS IMPORTANT**

### **Real-World Application:**
- Automatically categorize bug reports
- Route bugs to correct teams
- Prioritize bugs based on type
- Save time for software developers

### **Skills Demonstrated:**
1. **Data Cleaning**: Handling messy real-world data
2. **Feature Engineering**: Creating useful features
3. **Preprocessing**: Converting data for models
4. **EDA**: Understanding data through visualizations
5. **Model Comparison**: Trying multiple algorithms
6. **Evaluation**: Using proper metrics and validation
7. **Advanced Techniques**: SMOTE, Ensemble, Neural Networks

---

## **FINAL RESULTS COMPARISON**

| Model | Accuracy | Notes |
|-------|----------|-------|
| **Stacking Classifier** | **94.34%** | Best - combines multiple models |
| Random Forest | 91.73% | Very good, single model |
| Decision Tree | 83.81% | Good but simpler |
| SVM | 81.53% | Good for high-dimensional data |
| Logistic Regression | 80.98% | Simple linear model |
| MLP Neural Network | 75.62% | Non-linear, could improve with tuning |
| KNN | 81.63-82.70% | Distance-based, depends on K |
| Gradient Boosting | 78.40% | Sequential learning |
| Gaussian Naive Bayes | 62.51% | Too simple for this problem |
| Perceptron | 33.33% | Too simple, needs more layers |

**Best Model:** Stacking Classifier (94.34% accuracy)

---

## **GLOSSARY OF TERMS**

- **Dataframe**: Table-like data structure (pandas)
- **Feature**: An input variable (like Summary, Product)
- **Target/Label**: What we want to predict (Bug Type)
- **Overfitting**: Model memorizes training data too well
- **Underfitting**: Model too simple to learn patterns
- **Hyperparameter**: Setting you choose (like K in KNN)
- **Parameter**: Value model learns (like weights in neural network)
- **Cross-validation**: Testing model on multiple train/test splits
- **Confusion Matrix**: Table showing prediction vs actual labels
- **Precision**: Of predictions, how many are correct?
- **Recall**: Of actual cases, how many did we find?
- **F1 Score**: Balance of precision and recall
- **Ensemble**: Combining multiple models
- **Gradient Descent**: Algorithm to minimize error by adjusting weights
- **Backpropagation**: Calculating gradients in neural networks

---
