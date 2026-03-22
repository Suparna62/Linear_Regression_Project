# 📊 Linear Regression as a Modelling Tool (Metallurgy Project)

## 🚀 Overview
This project demonstrates the use of **Linear Regression** to model the relationship between **Ultimate Strength (Su)** and **Yield Strength (Sy)** using a real metallurgical dataset.

The goal is to understand:
- Choice of variables  
- Model behavior  
- Effect of hyperparameters  

---

## 🎯 Objective
- Build a simple linear regression model  
- Fit a linear relationship between input and output  
- Analyze the effect of hyperparameters:
  - Learning Rate  
  - Batch Size  
  - Number of Epochs  

---

## 📂 Dataset
- File: `Data.csv`  
- Domain: Metallurgy (Mechanical Properties of Materials)

### 🔑 Variables Used:
- **Input (x):** Ultimate Strength (Su)  
- **Output (y):** Yield Strength (Sy)  

---

## 🧠 Model
The linear regression model is: y = mx + c

Where:
- `m` = slope  
- `c` = intercept  

---

## ⚙️ Implementation

### 🔹 Libraries Used:
- pandas  
- numpy  
- matplotlib  
- scikit-learn  

### 🔹 Steps:
1. Load and clean dataset  
2. Convert data to numeric format  
3. Select input (Su) and output (Sy)  
4. Train model using sklearn  
5. Implement Gradient Descent manually  
6. Perform hyperparameter analysis  

---

## 📊 Results
- The model finds a strong linear relationship between **Su and Sy**  
- Output includes:
  - Regression plot  
  - Model equation  

Example: y = mx + c
---

## 🔬 Hyperparameter Analysis

### Parameters Tested:
- Learning Rate  
- Batch Size  
- Epochs  

### Observations:
- High learning rate → unstable  
- Low learning rate → slow convergence  
- Moderate learning rate → best performance  
- Larger batch size → smoother updates  

---

## 📈 Visualization
- Scatter plot (Actual Data)  
- Regression line  
- Learning rate comparison  

---

## 🧾 Conclusion
- Linear regression successfully models metallurgical behavior  
- Strong relationship between Ultimate Strength and Yield Strength  
- Hyperparameters significantly affect training performance  

---

## ▶️ How to Run

Install dependencies: ```pip install pandas numpy matplotlib scikit-learn```
Run the project: ``` python main.py```
