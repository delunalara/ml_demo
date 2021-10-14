import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt

import seaborn as sns
from sklearn.linear_model import LinearRegression


def anscombe_demo():
    data = sns.load_dataset('anscombe')
    data.iloc[:, 1:] = data.iloc[:, 1:].applymap(lambda x: np.around(x, 2))

    single = st.sidebar.checkbox('Single dataset', True)
    add_stats = st.sidebar.checkbox('Add descriptive statistics')
    add_plots = st.sidebar.checkbox('Add plots')
    add_dists = st.sidebar.checkbox('Add distribution')
    add_appropriate = st.sidebar.checkbox('Add appropriate model')

    if single:
        dataset = st.sidebar.select_slider('Dataset', data['dataset'].unique())
        
        with st.container():
            _, col, _ = st.columns((2.5, 3, 2))

            selected_dataset = data.loc[data['dataset'] == dataset, :]
            col.dataframe(selected_dataset.style.format('{:.2f}', subset=['x', 'y']))

            if add_stats:
                col.markdown('### Descriptive statistics')
                means = selected_dataset.iloc[:, 1:].agg(['mean', 'std']).apply(lambda x: np.around(x, 2))
                col.write(means.style.format('{:.2f}'))
                
                col.markdown('### Pearson correlation')
                col.write(selected_dataset.corr().iloc[1::2, :-1])

                model = LinearRegression(normalize=False)
                model.fit(selected_dataset[['x']], selected_dataset[['y']])
                r2 = model.score(selected_dataset[['x']], selected_dataset[['y']])

                col.markdown(f'### Linear Regression ($R^2 = {r2:.2f}$)')
                col.markdown(f'$y = {model.coef_[0][0]:.2f}x {"+" if model.intercept_ >= 0 else ""}{model.intercept_[0]:.2f}$')
        
        if add_plots:
            col.markdown('### Scatter plot')
            selected_dataset['pred'] = model.predict(selected_dataset[['x']])

            scatter_plot = alt.Chart(selected_dataset).mark_point().encode(
                x=alt.X('x', scale=alt.Scale(domain=[3, 20])),
                y=alt.Y('y', scale=alt.Scale(domain=[3, 15]))
            ) + alt.Chart(selected_dataset).mark_line().encode(
                x=alt.X('x', scale=alt.Scale(domain=[3, 20])),
                y=alt.Y('pred', scale=alt.Scale(domain=[3, 15]), title='y')
            )

            if add_appropriate:
                if dataset == 'I':
                    selected_dataset['pred_best'] = selected_dataset['pred']
                if dataset == 'II':
                    from sklearn.preprocessing import PolynomialFeatures
                    from sklearn.pipeline import make_pipeline 
                    best_model = make_pipeline(
                        PolynomialFeatures(degree=2),
                        LinearRegression(normalize=False)
                    )
                    best_model.fit(selected_dataset[['x']], selected_dataset[['y']])
                    selected_dataset['pred_best'] =  best_model.predict(selected_dataset[['x']])
                if dataset == 'III':
                    from sklearn.linear_model import TheilSenRegressor
                    best_model = TheilSenRegressor()
                    best_model.fit(selected_dataset[['x']], selected_dataset[['y']])
                    selected_dataset['pred_best'] =  best_model.predict(selected_dataset[['x']])
                if dataset != 'IV':
                    scatter_plot = scatter_plot + alt.Chart(selected_dataset).mark_line(color='green').encode(
                        x=alt.X('x', scale=alt.Scale(domain=[3, 20])),
                        y=alt.Y('pred_best', scale=alt.Scale(domain=[3, 15]), title='y')
                    )

            col.altair_chart(
                scatter_plot
            )
            col.markdown('### Box plot')
            col.altair_chart(
                alt.Chart(selected_dataset).mark_boxplot(color='blue').encode(
                    x='y'
                ).properties(
                    height=50
                )
            )
        if add_dists:
            col.markdown('### Distribution')
            col.altair_chart(
                alt.Chart(selected_dataset).transform_density(
                    'y', 
                    as_=['y', 'density'],
                    bandwidth=1
                ).mark_area(opacity=0.5).encode(
                    x='y',
                    y='density:Q'
                )
            )         

    else:
        with st.container():
            _, col, _  = st.columns((1, 4, 1))
            
            display_df = []
            for name, group in data.groupby('dataset'):
                group = group.reset_index(drop=True)
                group.columns = [f"{col}_{name}" for col in group.columns]
                display_df.append(group.iloc[:, 1:])

            col.write(pd.concat(display_df, axis=1))

            if add_stats:
                col.markdown('### Descriptive statistics')
                means = data.groupby('dataset').agg(['mean', 'std']).apply(lambda x: np.around(x, 2))
                col.write(means.style.format('{:.2f}'))

                col.markdown('### Pearson correlation')
                col.write(data.groupby('dataset').corr().iloc[1::2, :-1].applymap(lambda x: np.around(x, 3)).style.format('{:.2f}'))

                col.markdown(f'### Linear Regression ($y = mx + b$)')

            model_df = []
            models = {}
            for name, group in data.groupby('dataset'):
                model = LinearRegression(normalize=False)
                model.fit(group[['x']], group[['y']])
                models[name] = model
                model_df.append({
                    'dataset': name,
                    'm': np.around(model.coef_[0][0], 2),
                    'b': np.around(model.intercept_[0], 2),
                    'R^2': np.around(model.score(group[['x']], group[['y']]), 2)
                })
            if add_stats:
                col.write(pd.DataFrame(model_df).style.format('{:.2f}', subset=['m', 'b', 'R^2']))

        if add_plots:
            with st.container():
                charts = []
                _, col, _ = st.columns((0.5, 3, 0.5))
                for name, group in data.groupby('dataset'):
                    group['pred'] = models[name].predict(group[['x']])
                    scatter_plot = alt.Chart(group).mark_point().encode(
                        x=alt.X('x', scale=alt.Scale(domain=[3, 20])),
                        y=alt.Y('y', scale=alt.Scale(domain=[3, 15]))
                    ) + alt.Chart(group).mark_line().encode(
                        x=alt.X('x', scale=alt.Scale(domain=[3, 20])),
                        y=alt.Y('pred', scale=alt.Scale(domain=[3, 15]), title='y')
                    )
                    if add_appropriate:
                        titles = {
                            'I': 'Linear regression',
                            'II': 'Polynomial regression',
                            'III': 'Robust regression (Theil-Sen)',
                            'IV': 'None'
                        }
                        if name == 'I':
                            group['pred_best'] = group['pred']
                        if name == 'II':
                            from sklearn.preprocessing import PolynomialFeatures
                            from sklearn.pipeline import make_pipeline 
                            best_model = make_pipeline(
                                PolynomialFeatures(degree=2),
                                LinearRegression(normalize=False)
                            )
                            best_model.fit(group[['x']], group[['y']])
                            group['pred_best'] =  best_model.predict(group[['x']])
                        if name == 'III':
                            from sklearn.linear_model import TheilSenRegressor
                            best_model = TheilSenRegressor()
                            best_model.fit(group[['x']], group[['y']])
                            group['pred_best'] =  best_model.predict(group[['x']])
                        if name != 'IV':
                            scatter_plot = scatter_plot + alt.Chart(group).mark_line(color='green').encode(
                                x=alt.X('x', scale=alt.Scale(domain=[3, 20])),
                                y=alt.Y('pred_best', scale=alt.Scale(domain=[3, 15]), title='y')
                            )
                    charts.append(
                        scatter_plot.properties(
                            title=(['Model: Linear regression'] if not add_appropriate 
                                    else ['Model: Linear regression'] + [f'Best model: {titles[name]}'])
                        )
                    )
                col.markdown('### Scatter plots')
                col.altair_chart(
                    alt.vconcat(
                        alt.hconcat(charts[0], charts[1]),
                        alt.hconcat(charts[2], charts[3])
                    ).properties(
                    )
                )

                col.markdown('### Box plots')
                col.altair_chart(
                    alt.Chart(data).mark_boxplot(color='blue').encode(
                        x='y',
                        y='dataset'
                    ).properties(
                        height=300,
                        width=700,
                        padding={'left': 200}
                    )
                )
        if add_dists:
            with st.container():
                charts = []
                _, col, _ = st.columns((0.5, 3, 0.5))
                for name, group in data.groupby('dataset'):
                    charts.append(
                        alt.Chart(group).transform_density(
                            'y', 
                            as_=['y', 'density'],
                            bandwidth=1
                        ).mark_area(opacity=0.5).encode(
                            x='y',
                            y='density:Q'
                        )
                    )
                col.markdown('### Distribution plots')
                col.altair_chart(
                    alt.vconcat(
                        alt.hconcat(charts[0], charts[1]),
                        alt.hconcat(charts[2], charts[3])
                    ).properties(
                    )
                )
