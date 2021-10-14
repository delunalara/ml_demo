import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt

from time import sleep
from sklearn.datasets import make_blobs

def gini_impurity(*probabilities):
    out = 0
    for p in probabilities:
        out += p * (1 - p)
    return out

def decision_tree_demo():
    sub_demo = st.selectbox('', ['Split selection'], 0)

    if sub_demo == 'Split selection':
        dataset = st.select_slider('Dataset', [1, 2])
        if dataset == 1:
            X, y = make_blobs(centers=2, random_state=2)
            domain_x = [-7, 7]
            domain_y = [-14, 4]
        if dataset == 2:
            X, y = make_blobs(centers=2, cluster_std=2, random_state=6)
            domain_x = [-3, 15]
            domain_y = [-20, 5]
        df = pd.DataFrame(X)
        df.columns = ['x0', 'x1']
        df['y'] = y

        base = alt.Chart(df).mark_point().encode(
            x=alt.X('x0', scale=alt.Scale(domain=domain_x)),
            y=alt.Y('x1', scale=alt.Scale(domain=domain_y)),
            color='y:O'
        ).properties(
            width=600,
            height=300
        )

        chart = st.empty()
        chart.altair_chart(
            base
        )
        
        if st.button('Run simulation'):
            current_edge = st.empty()
            current_gini = st.empty()
            st.markdown('Split 1')
            split1 = st.empty()
            split1_gini = st.empty()
            st.markdown('Split 2')
            split2 = st.empty()
            split2_gini = st.empty()

            splits = []
            
            for variable in ['x0', 'x1']:
                edges = np.histogram_bin_edges(df[variable], 10).tolist()
                edge_step = edges[1] - edges[0]
                edges = [edges[0] - edge_step] + edges + [edges[-1] + edge_step]
                for edge in edges:
                    df['bin'] = df[variable] >= edge
                    current_edge.markdown(f"Split value: `{edge:.3f}`")
                    split1_values = df.loc[df['bin'], 'y'].value_counts()
                    split1.dataframe(split1_values)
                    split1_p0 = (split1_values.loc[0] if 0 in split1_values.index else 0) / split1_values.sum()
                    split1_p1 = (split1_values.loc[1] if 1 in split1_values.index else 0) / split1_values.sum()
                    split1_gini.markdown('Gini impurity = $$%0.2f \\times (1 - %0.2f) + %0.2f \\times (1 - %0.2f) = %0.2f$$' % (
                        split1_p0,
                        split1_p0,
                        split1_p1,
                        split1_p1,
                        gini_impurity(split1_p0, split1_p1)
                    ))

                    split2_values = df.loc[~df['bin'], 'y'].value_counts()
                    split2.dataframe(split2_values)
                    split2_p0 = (split2_values.loc[0] if 0 in split2_values.index else 0) / split2_values.sum()
                    split2_p1 = (split2_values.loc[1] if 1 in split2_values.index else 0) / split2_values.sum()
                    split2_gini.markdown('Gini impurity = $$%0.2f \\times (1 - %0.2f) + %0.2f \\times (1 - %0.2f) = %0.2f$$' % (
                        split2_p0,
                        split2_p0,
                        split2_p1,
                        split2_p1,
                        gini_impurity(split2_p0, split2_p1)
                    ))

                    gini_split1_weigthed = np.nan_to_num((split1_values.sum() / df.shape[0]) * gini_impurity(split1_p0, split1_p1))
                    gini_split2_weigthed = np.nan_to_num((split2_values.sum() / df.shape[0]) * gini_impurity(split2_p0, split2_p1))
                    split_gini = gini_split1_weigthed + gini_split2_weigthed
                    current_gini.markdown(f"Split Gini impurity: `{split_gini:0.3f}`")

                    if variable == 'x1':
                        line = alt.Chart(pd.DataFrame({'y': [edge]})).mark_rule(color='blue').encode(y=alt.Y('y', title='x1'))
                    else:
                        line = alt.Chart(pd.DataFrame({'x': [edge]})).mark_rule(color='blue').encode(x=alt.X('x', title='x0'))

                    chart.altair_chart(
                        base + line
                    )
                    sleep(0.25)

                    splits.append({'variable': variable, 'split': edge, 'gini': split_gini})

            splits_df = pd.DataFrame(splits).sort_values('gini')
            st.write(splits_df)

            if splits_df.iloc[0]['variable'] == 'x1':
                line = alt.Chart(pd.DataFrame({'y': [splits_df.iloc[0]['split']]})).mark_rule(color='green').encode(y=alt.Y('y', title='x1'))
            elif splits_df.iloc[0]['variable'] == 'x0':
                line = alt.Chart(pd.DataFrame({'x': [splits_df.iloc[0]['split']]})).mark_rule(color='green').encode(x=alt.X('x', title='x0'))

            chart.altair_chart(
                base + line
            )
    if sub_demo == 'Tree building':
        from sklearn.datasets import load_iris

        iris = load_iris()
        df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
        df['species'] = pd.Series(iris['target']).map({i: v for i, v in enumerate(iris['target_names'])})
        st.write(df)

        base = alt.Chart(df).mark_point().encode(
            x=alt.X('sepal length (cm)', scale=alt.Scale(domain=[4.0, 8.0])),
            y=alt.Y('petal length (cm)', scale=alt.Scale(domain=[0.0, 8.0])),
            color='petal width (cm)',
            shape='species'
        ).properties(
            width=700,
            height=400
        )

        chart = st.empty()
        chart.altair_chart(
            base
        )

        split_container = st.empty()

        step = 1
        if st.button('Run simulation'):
            if step == 1:
                splits = []
                
                for variable in iris['feature_names']:
                    edges = np.histogram_bin_edges(df[variable], 10).tolist()
                    edge_step = edges[1] - edges[0]
                    edges = [edges[0] - edge_step] + edges + [edges[-1] + edge_step]
                    for edge in edges:
                        df['bin'] = df[variable] >= edge

                        split1_values = df.loc[df['bin'], 'species'].value_counts()
                        split1_p0 = (split1_values.loc['setosa'] if 'setosa' in split1_values.index else 0) / split1_values.sum()
                        split1_p1 = (split1_values.loc['versicolor'] if 'versicolor' in split1_values.index else 0) / split1_values.sum()
                        split1_p2 = (split1_values.loc['virginica'] if 'virginica' in split1_values.index else 0) / split1_values.sum()

                        split2_values = df.loc[~df['bin'], 'species'].value_counts()
                        split2_p0 = (split2_values.loc['setosa'] if 'setosa' in split2_values.index else 0) / split2_values.sum()
                        split2_p1 = (split2_values.loc['versicolor'] if 'versicolor' in split2_values.index else 0) / split2_values.sum()
                        split2_p2 = (split2_values.loc['virginica'] if 'virginica' in split2_values.index else 0) / split2_values.sum()

                        gini_split1_weigthed = np.nan_to_num((split1_values.sum() / df.shape[0]) * gini_impurity(split1_p0, split1_p1, split1_p2))
                        gini_split2_weigthed = np.nan_to_num((split2_values.sum() / df.shape[0]) * gini_impurity(split2_p0, split2_p1, split2_p2))
                        split_gini = gini_split1_weigthed + gini_split2_weigthed

                        splits.append({'variable': variable, 'split': edge, 'gini': split_gini})

                splits_df = pd.DataFrame(splits).sort_values('gini')
                split_container.dataframe(splits_df)

