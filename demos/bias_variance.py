import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from mlxtend.evaluate import bias_variance_decomp
from mlxtend.plotting import plot_decision_regions


def bias_variance_demo():
    n_neighbors = st.slider('n_neighbors', 1, 10, 5, 1)

    loader = load_iris()
    X, y = loader['data'], loader['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # from sklearn.metrics import mean_squared_error
    # metrics = []
    # for d in range(1, 11):
    #    model = KNeighborsRegressor(d)
    #    mse, bias, variance = bias_variance_decomp(model, X_train, y_train, X_test, y_test, loss='mse', random_seed=1)
    #    model.fit(X_train, y_train)
    #    metrics.append({ 
    #        'bias': np.sqrt(bias), 
    #        'variance': variance, 
    #        'mse_train': mean_squared_error(y_train, model.predict(X_train)),
    #        'mse_test': mean_squared_error(y_test, model.predict(X_test))
    #     })
    # metrics_df = pd.DataFrame(metrics).reset_index()
    # metrics_df.rename(columns={'index': 'k'}, inplace=True)
    # metrics_df['k'] += 1
    # metrics_df.to_parquet('knn_bv.pt')

    pca = PCA(n_components = 2)
    X_2 = pca.fit_transform(X)
    X_2_train = pca.fit_transform(X_train)
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_2_train, y_train)

    plt.figure(figsize=(10, 4))
    plot_decision_regions(X_2, y.astype(np.int_), clf=model, legend=0)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    st.pyplot(plt.gcf())

    metrics_df = pd.read_parquet('knn_bv.pt')

    metrics_df_mse = metrics_df.loc[:, ['k', 'mse_train', 'mse_test']].melt(id_vars='k')

    error = alt.Chart(metrics_df_mse).mark_line().encode(
        x='k',
        y='value',
        color='variable'
    ).properties(
        width=650
    )

    base = alt.Chart(metrics_df).encode(
        alt.X('k:O', axis=alt.Axis(title='k'), scale=alt.Scale(reverse=True))
    ).properties(
        width=700
    )

    line1 = base.mark_line(color='#57A44C').encode(
        alt.Y('bias', axis=alt.Axis(title='bias', titleColor='#57A44C'))
    )

    line2 = base.mark_line(color='#5276A7').encode(
        alt.Y('variance', axis=alt.Axis(title='variance', titleColor='#5276A7'))
    )

    rule = alt.Chart(pd.DataFrame({'x': n_neighbors}, index=[0])).mark_rule(color='red').encode(
        x='x:O'
    )

    st.write(
        alt.layer(line1, line2 + rule).resolve_scale(
            y = 'independent'
        )
    )