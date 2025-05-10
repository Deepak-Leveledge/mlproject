## End to End Machine learning Project

📚 Student Performance Indicator
A machine learning-powered web application that predicts student performance—specifically math scores—based on various socio-economic, demographic, and academic factors. This project offers actionable insights for educators and institutions aiming to understand the key drivers behind student success.

📌 Project Overview
This project aims to build a robust regression model to predict a student's math score based on features such as parental education level, gender, lunch type, and test preparation course. It can be easily extended to predict other performance metrics like reading or writing scores by adjusting the target variable.

🧠 Objectives
Predict student math performance using regression techniques.

Identify key features that contribute to academic success.

Provide a user-friendly interface to input data and receive predictions.

Enable educators to make data-driven interventions.

📊 Dataset
Source: Student Performance Dataset (Kaggle or UCI)

Total Records: 1000+

Target Variable: Math Score (can be changed to Reading/Writing Score)

Features:

Gender

Race/Ethnicity

Parental level of education

Lunch (standard/reduced)

Test preparation course (completed/not)

Reading score

Writing score

🧪 Machine Learning Models
Regression Models Used
Linear Regression

Decision Tree Regressor

Random Forest Regressor

Gradient Boosting Regressor

Evaluation Metrics
Mean Absolute Error (MAE)

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

R² Score

🔧 Hyperparameter Tuning
Used GridSearchCV for fine-tuning models and selecting optimal parameters for better prediction accuracy.

🖥️ Tech Stack
Frontend: Flask + HTML5/CSS3

Backend: Python, Flask

Model Development: Jupyter, scikit-learn

Visualization: Matplotlib, Seaborn

🌐 Live Demo
🚀 Try it now: Student Performance Predictor on Render https://mlproject-pojk.onrender.com/
🔄 Flexibility
This system is built to predict math scores, but you can easily modify the target variable to predict:

Reading Score

Writing Score

Overall GPA (if data allows)

Just change the label column in model.py and retrain the model.

🛠️ Libraries Used
pandas

numpy

scikit-learn

matplotlib

seaborn

Flask

Jupyter Notebook
