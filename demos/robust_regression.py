import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt

from sklearn.linear_model import LinearRegression
from time import sleep

def generate(median, noise, outlier_noise, n_samples, n_outliers):
        errs = noise * np.random.rand(n_samples) * np.random.choice((-1, 1), n_samples)
        data = median + errs
        
        lower_errs = outlier_noise * np.random.rand(n_outliers)
        lower_outliers = median - noise - lower_errs

        upper_errs = outlier_noise * np.random.rand(n_outliers)
        upper_outliers = median + noise + upper_errs

        data = np.concatenate((data, lower_outliers, upper_outliers))
        np.random.shuffle(data)

        return data

def theil_sen_demo():
    n_samples = st.sidebar.slider('n_samples', 10, 100, 20, 10)
    noise = st.sidebar.slider('noise', 1.0, 20.0, 3.0, 0.1)
    outlier_noise = st.sidebar.slider('outlier_noise', 50, 200, 50, 10)
    percent_outliers = st.sidebar.slider('percent_outliers', 0, 50, 25, 1, format='%d%%')      

    samples = int(n_samples - n_samples * (percent_outliers / 100))
    data = generate(median=np.random.normal(0, 100), noise=noise, outlier_noise=outlier_noise, n_samples=samples, n_outliers=n_samples-samples)
    df = pd.DataFrame(data).reset_index()
    X = df['index'].values
    y = df[0].values

    df = pd.DataFrame((np.squeeze(X), y)).rename(index={0: 'X', 1: 'y'}).T
    df = df.sort_values('X').reset_index(drop=True)
    st.markdown('### Data')
    st.write(df)

    base = alt.Chart(df).mark_point().encode(
        x='X',
        y='y'
    ).properties(
        width=600
    )

    chart_row = st.empty()
    chart_row.altair_chart(base)

    speed = st.sidebar.slider('speed', 10, 120, 60, 5)
    st.markdown('### Estimation')
    cur_median = st.empty()
    cur_median.markdown(r'Estimated slope = `NaN`')
    cur_intercept = st.empty()
    cur_intercept.markdown(r'Estimated intercept = `NaN`')

    slope_calc = st.empty()
    slope_calc.markdown(r'Current slope = `NaN`')

    chart_df = st.empty()
    pts_df = pd.DataFrame(columns=['p1', 'p2', 'm'])
    chart_df.dataframe(pts_df)

    run = st.sidebar.button('Start simulation')   

    if run:
        from itertools import combinations
        pairs = list(combinations(df.index.values, 2))
        for pair in pairs:
            sleep(1/speed)
            chart_row.altair_chart(
                base +
                alt.Chart(df.loc[pair, :]).mark_line(color='red').encode(
                    x='X',
                    y='y'
                )
            )
            new_row = pd.DataFrame({
                'p1': pair[0],
                'p2': pair[1],
                'm': (df.loc[pair[1], 'y'] - df.loc[pair[0], 'y'])/(df.loc[pair[1], 'X'] - df.loc[pair[0], 'X'])
            }, index=[0])
            pts_df = pd.concat((new_row, pts_df), ignore_index=True, axis=0)
            chart_df.dataframe(pts_df)
            cur_median.markdown(f'Estimated slope = `{pts_df["m"].median():.2f}`')
            slope_calc.markdown(f'Current slope = `{pts_df.loc[0, "m"]:.2f}`')
            slope = pts_df['m'].median()
            intercept = np.median(y) - slope * np.median(X)           
            cur_intercept.markdown(f'Estimated intercept = `{intercept:.2f}`')
        df['pred'] = (df['X'] * slope) + intercept

        from sklearn.linear_model import TheilSenRegressor
        model = TheilSenRegressor()
        model.fit(df[['X']], df[['y']])
        df['pred2'] = model.predict(df[['X']])

        sleep(5)
        chart_row.altair_chart(
            base + alt.Chart(df).mark_line().encode(
                x='X',
                y='pred'
            ) + alt.Chart(df).mark_line(color='green').encode(
                x='X',
                y='pred2'
            )
        )  

def ransac_demo():
    n_samples = st.sidebar.slider('n_samples', 10, 100, 20, 10)
    noise = st.sidebar.slider('noise', 1.0, 20.0, 3.0, 0.1)
    outlier_noise = st.sidebar.slider('outlier_noise', 50, 200, 50, 10)
    percent_outliers = st.sidebar.slider('percent_outliers', 0, 50, 25, 1, format='%d%%')      

    samples = int(n_samples - n_samples * (percent_outliers / 100))
    data = generate(median=np.random.normal(0, 100), noise=noise, outlier_noise=outlier_noise, n_samples=samples, n_outliers=n_samples-samples)
    df = pd.DataFrame(data).reset_index()
    X = df['index'].values
    y = df[0].values

    df = pd.DataFrame((np.squeeze(X), y)).rename(index={0: 'X', 1: 'y'}).T
    df = df.sort_values('X').reset_index(drop=True)
    st.markdown('### Data')
    st.write(df)

    base = alt.Chart(df).mark_point().encode(
        x='X',
        y='y'
    ).properties(
        width=600
    )

    chart_row = st.empty()
    chart_row.altair_chart(base)

    max_it = st.sidebar.slider('max_it', 10, 100, 50, 1)
    speed = st.sidebar.slider('speed', 10, 120, 60, 5)

    st.markdown('### Estimation')
    rmse_text = st.empty()
    rmse_text.markdown(r'Current rmse: `NaN`\t\t\t\tBest rmse: `NaN`')        

    run = st.sidebar.button('Start simulation')       
    
    if run:
        from sklearn.metrics import mean_squared_error
        from sklearn.linear_model import RANSACRegressor
        best_rmse_ = np.inf
        best_inliers_ = 0
        best_model = None
        for i in range(max_it):
            sample = df.sample(n_samples // 10)
            model = LinearRegression()
            model.fit(sample[['X']], sample[['y']])
            df['y_pred'] = model.predict(df[['X']])
            residual_threshold = np.median(np.abs(df['y_pred'] - np.median(df['y_pred'])))
            df['residual'] = np.abs(df['y'] - df['y_pred'])
            df['inlier'] = df['residual'] <= residual_threshold
            inliers = df['inlier'].sum()
            chart_row.altair_chart(
                alt.Chart(df).mark_point().encode(
                    x='X',
                    y='y'
                ) +
                alt.Chart(df).mark_line(width=i).encode(
                    x='X',
                    y='y_pred'
                ).properties(
                    width=600
                )
            )
            new_sample = pd.concat((sample, df.loc[df['inlier']])).drop_duplicates()
            better_model = LinearRegression()
            better_model.fit(new_sample[['X']], new_sample[['y']])
            new_sample['y_pred'] = better_model.predict(new_sample[['X']])
            rmse = np.sqrt(mean_squared_error(new_sample['y'], new_sample['y_pred']))
            if rmse < best_rmse_:
                best_model = better_model
                best_rmse_ = rmse
            rmse_text.markdown(f'Current rmse: `{rmse:.3f}`\t\t\t\tBest rmse: `{best_rmse_:.3f}`')
            sleep(1 / speed)
        sleep(2)
        df['y_pred'] = best_model.predict(df[['X']])
        chart_row.altair_chart(
            alt.Chart(df).mark_point().encode(
                x='X',
                y='y'
            ) +
            alt.Chart(df).mark_line(color='green').encode(
                x='X',
                y='y_pred'
            ).properties(
                width=600
            )
        )
        rmse_text.markdown(f'Current rmse: `{best_rmse_:.3f}`\t\t\t\tBest rmse: `{best_rmse_:.3f}`')