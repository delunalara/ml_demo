import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt

from sklearn.datasets import make_regression

def least_squares_demo():
    st.markdown('## Ordinary Least Squares')
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
        min_value=0, 
        max_value=n_features, 
        value=n_features, 
        step=1
    ) if n_features > 1 else None
    if effective_rank is not None and effective_rank == 0:
        effective_rank = 1e-100

    if explain:
        st.markdown("""
        ***
        This demo shows how to solve for the coefficients of a linear regression using 
        linear algebra. The task is to find a vector $\\beta$ of coefficients that 
        minimizes loss given a feature matrix $X$ and a response vector $y$.

        The general equation of a line is given by $y = m \\cdot x + b$ where $m$ is the
        slope and $b$ is the intercept. We can express this equation in linear algebra as
        $y = \\beta_1 \\cdot X + \\beta_0$.

        In this notation, $y$ is a response vector of shape $n \\times 1$, where $n$ is 
        the number of observations; and $X$ is the feature matrix or design matrix of 
        shape $n \\times m$, where $m$ is the number of features.

        $\\beta_0$ and $\\beta_1$ can be simplified considering that the intercept can be
        embedded in the feature matrix as a constant, yielding this form of the equation:

        $y = \\beta X$
        
        Where $\\beta$ is the vector of coefficients of shape $m + 1 \\times 1$ (i.e. it
        contains the intercept). We can then solve for $\\beta$ using linear algebra.

        To clear $X$ we have to get its inverse, but first we have to make it a square 
        matrix by multiplying by its transpose on both sides of the equation:

        $X^Ty = \\beta X^TX$

        We can then multiply the inverse of this product (in terms of $X$ )on both sides:

        $\\left(X^TX\\right)^{-1}X^Ty = \\beta X^TX\\left(X^TX\\right)^{-1}$ 

        A matrix times its inverse yields the identity matrix ($AA^{-1} = I$) so we can
        remove all $X$ terms from the right, yielding the final form of the equation:

        $\\beta = \\left(X^TX\\right)^{-1}X^Ty$

        This is the form of Ordinary Least Squares in linear algebra.
        ***
        """)

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

    corr_df = pd.DataFrame(X).iloc[:, :-1].corr()
    corr_df.index = [f'x{i}' for i in range(n_features)]
    corr_df.columns = [f'x{i}' for i in range(n_features)]
    st.markdown('Correlation (Pearson)')
    if explain:
        st.markdown(
            """
            In OLS, features are assumed to be independent, if a feature is function 
            of others, then the matrix is not directly invertible. Features with a
            Pearson correlation coefficient higher than 0.75 can be considered highly
            dependent.

            Move the `effective_rank` slider and see how these values change.
            """
        )
    st.write(corr_df.style.text_gradient(cmap='RdYlGn', vmin=-1.0, vmax=1.0))
    st.markdown('***')

    df = pd.DataFrame(X)
    df.columns = [f'x{i}' for i in range(n_features)] + ['intercept']
    df['y'] = y

    st.markdown('$\{X, y\}$')
    if explain:
        st.markdown(
            """
            This shows the set of values of $X$ and $y$, notice the extra column
            added by the intercept, which is a vector of ones.
            """
        )
    st.write(df)
    st.markdown('***')

    st.markdown('$X^T$')
    if explain:
        st.markdown(
            """
            This is the transpose of the feature/design matrix $X$, which switches rows
            and columns, implemented by the `.T` operator: `X.T`
            """
        )
    st.write(X.T)
    st.markdown('***')

    st.markdown('$X^TX$')
    if explain:
        st.markdown(
            """
            Notice how the matrix is square. A matrix is said to be invertible if it is
            square and its columns are linearly independent. This matrix multiplication
            is implemented with the `np.matmul` operator: `np.matmul(X.T, X)`
            """
        )
    st.write(np.matmul(X.T, X))
    st.markdown("***")
    
    st.markdown(r'$\left(X^TX\right)^{-1}$')
    if explain:
        st.markdown(
            """
            This is the pseudoinverse (`np.linalg.pinv`) to handle low rank feature 
            matrices: `np.linalg.pinv(np.matmul(X.T, X))`
            """
        )
    st.write(np.linalg.pinv(np.matmul(X.T, X)))
    st.markdown('***')

    st.markdown('$X^Ty$')
    if explain:
        st.markdown(
            """
            `np.matmul(X.T, y)`
            """
        )
    st.write(np.matmul(X.T, y))
    st.markdown('***')

    st.markdown(r'$\beta = \left(X^TX\right)^{-1}X^Ty$')
    if explain:
        st.markdown(
            """
            Given all of the pieces calculated above, we can solve for $\\beta$:

            `np.matmul(np.linalg.pinv(np.matmul(X.T, X)), np.matmul(X.T, y))`
            
            Move the sliders on the sidebar to see how the values change. The `bias`
            corresponds to the intercept. Moving `n_informative` should decrease the
            magnitude of the coefficients.
            """
        )
    beta = np.matmul(np.linalg.pinv(np.matmul(X.T, X)), np.matmul(X.T, y))
    beta_df = pd.DataFrame(beta, index=[f'beta_{i}' for i in range(n_features)] + ['intercept'])
    st.write(beta_df)
    st.write('***')

    if explain:
        st.markdown(
            """
            These plots show each feature vs the response
            """
        )
    y_pred = np.dot(df.iloc[:, :-1], beta)

    pred_df = []
    for i in range(n_features):
        pred_df.append(np.dot(df.iloc[:, i], beta[i]))
    pred_df = pd.DataFrame(pred_df)

    if n_features > 1:
        fig, axs = plt.subplots(nrows=n_features, ncols=1, sharex=False, sharey=True, figsize=(7, 1.5 * n_features))
    else:
        fig = plt.figure(figsize=(7, 1.5))
        axs = [plt.gca()]
    for i in range(n_features):
        axs[i].scatter(df.iloc[:, i], df.iloc[:, -1], s=10, facecolor='royalblue')
        axs[i].plot(df.iloc[:, i], pred_df.iloc[i, :], color='grey', lw=0.5)
        axs[i].set_xlabel(f'x{i}')
        axs[i].set_ylabel(f'y')
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['top'].set_visible(False)
    plt.tight_layout(w_pad=0.5, h_pad=0.5)
    st.pyplot(fig)

    rmse = np.sqrt(np.sum(np.power(y - y_pred, 2)) / len(y))