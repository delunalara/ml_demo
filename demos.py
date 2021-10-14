import streamlit as st

from demos.least_squares import least_squares_demo
from demos.bias_variance import bias_variance_demo
from demos.gradient_descent import gradient_descent_demo
from demos.robust_regression import theil_sen_demo
from demos.robust_regression import ransac_demo
from demos.decision_tree import decision_tree_demo

demo_options = [
    'Decision Tree',
    'Least Squares',
    'Gradient Descent', 
    'Theil-Sen', 
    'RANSAC', 
    'Bias/Variance Tradeoff (kNN)',
    'Towers of Hanoi'
]

selected = st.sidebar.selectbox('Demonstration', demo_options)

if selected == 'Least Squares':
    least_squares_demo()
    
if selected == 'Bias/Variance Tradeoff (kNN)':
    bias_variance_demo()

if selected == 'Theil-Sen':
    theil_sen_demo()

if selected == 'RANSAC':
    ransac_demo()
            
if selected == 'Gradient Descent':
    gradient_descent_demo()

if selected == 'Decision Tree':
    decision_tree_demo()
