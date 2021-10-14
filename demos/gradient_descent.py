import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt

from sklearn.datasets import make_regression

def gradient_descent_demo():
    st.markdown('## Gradient Descent')
    explain = st.sidebar.checkbox('Display explanation', True)
    
    if explain:
        st.sidebar.markdown('***')
        st.sidebar.markdown('Number of features in $X$ ($m$)')
    n_features = st.sidebar.slider(
        label='n_features', 
        min_value=1, 
        max_value=5, 
        value=1, 
        step=1
    )

    if explain and n_features > 1:
        st.sidebar.markdown('***')
        st.sidebar.markdown('Number of informative features '
            '(that are relevant to the regression)')
    n_informative = st.sidebar.slider(
        label='n_informative', 
        min_value=1, 
        max_value=n_features, 
        value=n_features, 
        step=1
    ) if n_features > 1 else 1

    if explain:
        st.sidebar.markdown('***')
        st.sidebar.markdown('Noise added to the response (irreducible error)')
    noise = st.sidebar.slider(
        label='noise', 
        min_value=0, 
        max_value=100, 
        value=0, 
        step=1
    )

    if explain:
        st.sidebar.markdown('***')
        st.sidebar.markdown('Constant for the regression (intercept)')
    bias = st.sidebar.slider(
        label='bias', 
        min_value=0.0, 
        max_value=10.0, 
        value=0.0, 
        step=0.01
    )

    if explain and n_features > 1:
        st.sidebar.markdown('***')
        st.sidebar.markdown('Rank of the feature matrix (independent features)')
    effective_rank = st.sidebar.slider(
        label='effective_rank', 
        min_value=0.0, 
        max_value=float(n_features), 
        value=float(n_features), 
        step=0.01
    ) if n_features > 1 else None
    if effective_rank is not None and effective_rank == 0:
        effective_rank = 1e-100

    if explain:
        st.sidebar.markdown('***')
        st.sidebar.markdown('Learning rate')
    alpha = st.sidebar.slider(
        label='alpha', 
        min_value=0.1, 
        max_value=1.0, 
        value=0.1, 
        step=0.01
    )

    if explain:
        st.sidebar.markdown('***')
        st.sidebar.markdown('Maximum number of iterations')
    max_it = st.sidebar.slider(
        label='max_it', 
        min_value=100, 
        max_value=1000, 
        value=100, 
        step=10
    )

    if explain:
        st.sidebar.markdown('***')
        st.sidebar.markdown('Optimization tolerance')
    tolerance = st.sidebar.select_slider(
        label='tolerance', 
        options=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1], 
        value=1e-3, 
        format_func=lambda s: f'{s:.2e}'
    )

    X, y = make_regression(
        n_samples=100, 
        n_features=n_features, 
        n_informative=n_informative, 
        noise=noise, 
        random_state=42,
        bias=bias,
        effective_rank=effective_rank,
        shuffle=False
    )
    X = np.hstack((X, np.ones((X.shape[0], 1))))

    if n_features == 1:
        df = pd.DataFrame(X, columns=['x', 'bias'])
        df['y'] = y
        df = df.sort_values('x')
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

    y = y.reshape(-1, 1)

    df = pd.DataFrame((X[:, 0], np.squeeze(y)), index=['x', 'y']).T

    if explain:
        st.markdown(
            """
            A cost function is used to minimize the least square loss as defined by
            the square difference between the actual and predicted values:

            $\\frac{1}{2n} \\cdot \\sum \\left(X \\theta - y\\right)^2$

            $\\theta$ is initialized with small random non-zero values and predicted
            values are calculated as the dot product of $X$ and $\\theta$

            Gi

            """
        )

    base = alt.Chart(df).mark_point().encode(
        x='x',
        y='y'
    ).properties(
        height=300,
        width=700
    )

    row_chart = st.empty()
    row_chart.altair_chart(
        base
    )

    cost_chart = st.empty()

    def gradient_descent(X, y, theta, alpha=0.01, max_it=100):
        def cost(theta, X, y):
            return (1/2 * len(y)) * np.sum(np.square(X.dot(theta) - y))

        cost_array = np.zeros(max_it)
        theta_array = np.zeros((max_it, n_features + 1))
        for it in range(max_it):
            current_it.markdown(f'Current iteration: `{it}`')
            prediction = np.dot(X,theta)
            pred_df = pd.DataFrame((X[:, 0], np.squeeze(prediction)), index=['x', 'y']).T
            if n_features == 1:
                row_chart.altair_chart(
                    base +
                    alt.Chart(pred_df).mark_line(color='green').encode(
                        x='x',
                        y='y'
                    )
                )
            theta = theta - (1/len(y)) * alpha * (X.T.dot((prediction - y)))
            st.write((X.T.dot((prediction - y))), np.sum(np.square(X.dot(theta) - y)))
            theta_text.dataframe(theta)
            theta_array[it,:] = theta.T
            cost_array[it]  = cost(theta,X,y)
            current_cost.markdown(f'Current cost: `{cost_array[it]:.2f}`')
            chart_data = pd.DataFrame(cost_array[:it]).reset_index().rename(columns={'index': 'iteration', 0: 'cost'})
            cost_chart.altair_chart(
                alt.Chart(chart_data).mark_line().encode(
                    x='iteration',
                    y='cost'
                ).properties(
                    height=300,
                    width=700
                )
            )
            cost_update = cost_array[it] - cost_array[it - 1]
            current_diff.markdown(f'Cost update: `{cost_update:.4f}`')
            if np.around(cost_array[it], 3) == 0.0 or np.abs(cost_update) < tolerance:
                break
            
        return theta, cost_array, theta_array

    theta = np.random.randn(n_features + 1, 1)

    current_it = st.empty()
    current_it.markdown('Current iteration: `NaN`')

    current_cost = st.empty()
    current_cost.markdown('Current cost: `NaN`')

    current_diff = st.empty()
    current_diff.markdown('Cost update: `NaN`')

    st.markdown('Current parameters')
    theta_text = st.empty()
    theta_text.dataframe(theta)

    run = st.button('Run simulation')
    if run:
        theta, cost_history, theta_history = gradient_descent(X, y, theta, alpha=alpha, max_it=max_it)
