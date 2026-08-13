Student Performance Prediction Using Machine Learning
About the Project

This project focuses on predicting student performance using Machine Learning. The aim is to use student-related information and build a model that can predict whether a student is likely to Pass or Fail.

The project was developed as part of my Data Science internship and learning journey, with the main focus on understanding how a Machine Learning project is developed from data preprocessing to model evaluation.

Dataset

The project uses a Student Performance dataset containing:

395 student records
33 features

The dataset includes information related to students' academic performance, study habits, family background, support systems, absences, and other factors.

Some important features include:

Age
Gender
Study time
Previous failures
Parental education
Family support
School support
Absences
G1 and G2 grades
Target Variable

The original dataset contains the G3 column, which represents the final grade.

For this project, a new Result column was created:

G3 >= 10  → Pass (1)
G3 < 10   → Fail (0)

This converts the problem into a binary classification problem.

Project Workflow
Dataset
   ↓
Data Understanding
   ↓
Data Preprocessing
   ↓
Target Variable Creation
   ↓
Categorical Data Encoding
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Model Comparison
Machine Learning Models

Three classification algorithms were implemented:

1. Logistic Regression

Used as a basic classification model and baseline for comparison.

2. Decision Tree

A tree-based model that makes predictions by splitting the data based on different features.

3. Random Forest

An ensemble model that combines multiple Decision Trees to make predictions.

Model Evaluation

The models were evaluated using:

Accuracy
Precision
Recall
F1-Score
Confusion Matrix

The performance of the three models was also compared using a visualization.

Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
VS Code
Project Structure
Student_Performance_Prediction/
│
├── dataset/
│   └── student_data.csv
│
├── main.py
│
├── README.md
│
└── requirements.txt
How to Run the Project
1. Clone the repository
git clone [https://github.com/shrutigaykar55-commits/Student-Performance-Prediction]
2. Open the project folder
cd Student_Performance_Prediction
3. Install the required libraries
pip install pandas numpy matplotlib seaborn scikit-learn
4. Run the project
python main.py

The program will train the three models, display their evaluation results, and generate the confusion matrices and model comparison graph.

Results

The final model results are compared using Accuracy, Precision, Recall and F1-Score.

Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	Add result	Add result	Add result	Add result
Decision Tree	Add result	Add result	Add result	Add result
Random Forest	Add result	Add result	Add result	Add result

Best Model: Add the model name after running the final code.

What I Learned

Working on this project helped me understand the practical workflow of a Machine Learning project, especially:

Working with a real dataset
Data preprocessing
Categorical data encoding
Train-test splitting
Classification algorithms
Model evaluation
Confusion matrices
Comparing Machine Learning models
Using Python libraries for Data Science
Future Improvements

Some possible improvements for the project are:

Using a larger dataset
Adding more relevant student-related features
Applying hyperparameter tuning
Testing additional Machine Learning algorithms
Building a simple web interface for predictions
