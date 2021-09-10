from altair.vegalite.v4.schema.channels import Y
from sklearn import preprocessing
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import altair as alt

# Regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# Pipelines
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Utility
from sklearn.model_selection import train_test_split

# Datasets
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
from sklearn.datasets import load_diabetes


def calc_height(n_cols):
    if n_cols <= 10:
        return 200
    return 15 * n_cols


problem_type = st.sidebar.selectbox(
    label='Problem type',
    options=[
        "Classification",
        "Regression"
    ]
)

if problem_type == 'Classification':
    algorithms = {
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "SVM"
    }
    datasets = {
        "iris": (load_iris, 'sklearn'),
        "digits": (load_digits, 'sklearn'),
        "wine": (load_wine, 'sklearn'),
        "breast_cancer": (load_breast_cancer, 'sklearn')
    }
elif problem_type == 'Regression':
    algorithms = {
        "Linear Regression",
        "Decision Tree",
        "kNN",
        "SVM"
    }
    datasets = {
        "boston": (load_boston, 'sklearn'),
        "diabetes": (load_diabetes, 'sklearn'),
        "tips": ('tips', 'seaborn'),
        "mpg": ('mpg', 'seaborn')
    }

dataset = st.sidebar.selectbox(
    label='Dataset',
    options=datasets
)

algorithm = st.sidebar.selectbox(
    label='Algorithm',
    options=algorithms
)

def load_dataset(args):
    def parse_description(description):
        description = re.sub(r'\.\. [a-z]+:: (.*)\n', '**\\1:**', description)
        description = re.sub(r'\.\. _[a-z_]+:', '', description)
        return description

    if args[1] == 'sklearn':
        loader = args[0]()
        data = pd.DataFrame(
            data=loader['data'], 
            columns=loader['feature_names']
        )
        if problem_type == 'Classification':
            data['class'] = pd.Series(loader['target']).map(
                {key:value for key, value in enumerate(loader['target_names'])}
            )
        elif problem_type == 'Regression':
            data['response'] = loader['target']
        description = parse_description(loader['DESCR'])
    if args[1] == 'seaborn':
        data = sns.load_dataset(args[0])
        if args[0] == 'tips':
            data['sex'] = (data['sex'] == 'Male').astype(int)
            data['smoker'] = (data['smoker'] == 'Yes').astype(int)
            data = data.drop('day', axis=1).join(pd.get_dummies(data['day'], prefix='day'))
            data = data.drop('time', axis=1).join(pd.get_dummies(data['time'], prefix='time'))
            data = data.rename(columns={'tip': 'response'})
        if args[0] == 'mpg':
            data = data.drop('name', axis=1)
            data = data.drop('origin', axis=1).join(pd.get_dummies(data['origin'], prefix='origin'))
            data = data.rename(columns={'mpg': 'response'})
            data = data.dropna()
        description = ''

    return data, description

data, description = load_dataset(datasets[dataset])
show_description = st.sidebar.checkbox('Show dataset description')
if show_description:
    st.write(description)

st.markdown('## Dataset Sample')
st.dataframe(data.sample(10, random_state=42))

if problem_type == 'Classification':
    y_key = 'class'
if problem_type == 'Regression':
    y_key = 'response'

data_train, data_test = train_test_split(
    data.reset_index(), 
    test_size=0.3, 
    stratify=data[y_key] if problem_type == 'Classification' else None, 
    random_state=42
)

data_train = data_train.set_index('index').sort_index()
data_test = data_test.set_index('index').sort_index()
X_train, y_train = data_train.drop(y_key, axis=1), data_train[y_key]
X_test, y_test= data_test.drop(y_key, axis=1), data_test[y_key]

if problem_type == 'Classification':
    pos_label = {
        'breast_cancer': 'malignant'
    }
    pos_label = pos_label[dataset] if dataset in pos_label else 1
    if algorithm == 'Logistic Regression':
        penalty = st.selectbox('Regularization', ['l1', 'l2', 'none'])
        if penalty != 'none':
            alpha = st.slider('alpha', 0.0, 1.0, 0.0, 0.01)
            alpha = 1e-10 if alpha == 0 else alpha
            model = LogisticRegression(penalty=penalty, solver='liblinear', C=1/alpha, random_state=42)
        else:
            model = LogisticRegression(penalty=penalty, solver='lbfgs', random_state=42)
    if algorithm == 'Decision Tree':
        max_depth = st.slider('max_depth', 1, 100, 10, 1)
        criterion = st.select_slider('criterion', ['gini', 'entropy'])
        splitter = st.select_slider('splitter', ['best', 'random'])
        min_samples_leaf = st.slider('min_samples_leaf', 1, 100, 10, 1)
        model = DecisionTreeClassifier(
            criterion=criterion, 
            splitter=splitter, 
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
    if algorithm == 'kNN':
        n_neighbors = st.slider('n_neighbors', 1, 100, 10, 1)
        weights = st.select_slider('weights', ['uniform', 'distance'])
        metric = st.select_slider('metric', ['euclidean', 'manhattan', 'chebyshev'])
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric
        )
    if algorithm == 'SVM':
        C = st.slider('C', 0.0, 1.0, 0.0, 0.01)
        C = 1e-10 if C == 0 else C
        kernel = st.select_slider('kernel', ['linear', 'rbf', 'sigmoid'])
        gamma = st.select_slider('gamma', ['auto', 'scale'])
        model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=True,
            random_state=42
        )

if problem_type == 'Regression':
    if algorithm == 'Linear Regression':
        base_model = LinearRegression(
            fit_intercept=True,
            normalize=False,
        )
        degree = st.slider('degree', 1, 5, 1, 1)
        regularization = st.select_slider('regularization', ['none', 'l1', 'l2'])
        if regularization == 'none':
            robust = st.select_slider('robust', ['none', 'TheilSen', 'RANSAC'])
            if robust == 'none':
                model = base_model
            if robust == 'TheilSen':
                model = TheilSenRegressor(random_state=42)
            if robust == 'RANSAC':
                model = RANSACRegressor(base_model, random_state=42)
        else:
            alpha = st.slider('alpha', 0.0, 1.0, 0.0, 0.01)
            if regularization == 'l1':
                model = Lasso(alpha=alpha, random_state=42)
            if regularization == 'l2':
                model = Ridge(alpha=alpha, random_state=42)
        model = make_pipeline(PolynomialFeatures(degree, include_bias=False), model)
    if algorithm == 'Decision Tree':
        max_depth = st.slider('max_depth', 1, 100, 10, 1)
        criterion = st.select_slider('criterion', ['mse', 'mae'])
        splitter = st.select_slider('splitter', ['best', 'random'])
        min_samples_leaf = st.slider('min_samples_leaf', 1, 100, 10, 1)
        model = DecisionTreeRegressor(
            criterion=criterion, 
            splitter=splitter, 
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
    if algorithm == 'kNN':
        n_neighbors = st.slider('n_neighbors', 1, 100, 10, 1)
        weights = st.select_slider('weights', ['uniform', 'distance'])
        metric = st.select_slider('metric', ['euclidean', 'manhattan', 'chebyshev'])
        model = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric
        )
    if algorithm == 'SVM':
        C = st.slider('C', 0.0, 1.0, 0.0, 0.01)
        C = 1e-10 if C == 0 else C
        kernel = st.select_slider('kernel', ['linear', 'rbf', 'sigmoid'])
        gamma = st.select_slider('gamma', ['auto', 'scale'])
        model = SVR(
            C=C,
            kernel=kernel,
            gamma=gamma
        )

model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

st.markdown('### Metrics')
if problem_type == 'Classification':
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import roc_auc_score

    if hasattr(model, 'decision_function'):
            y_proba_train = model.decision_function(X_train)
            y_proba_test = model.decision_function(X_test)
    if hasattr(model, 'predict_proba'):
        if y_train.nunique() == 2:
            y_proba_train = model.predict_proba(X_train)[:, 1]
            y_proba_test = model.predict_proba(X_test)[:, 1]
        else:
            y_proba_train = model.predict_proba(X_train)
            y_proba_test = model.predict_proba(X_test)

    train_metrics = {
        'accuracy': 
            np.around(accuracy_score(y_train, y_pred_train), 3), 
        'balanced_accuracy': 
            np.around(balanced_accuracy_score(y_train, y_pred_train), 3),
        f"f1_{'binary' if y_train.nunique() == 2 else 'weighted'}": 
            np.around(f1_score(y_train, y_pred_train, average='binary' if y_train.nunique() == 2 else 'weighted', pos_label=pos_label), 3),
        "roc_auc":
            np.around(roc_auc_score(y_train, y_proba_train, multi_class='ovr'), 4)
    }

    test_metrics = {
        'accuracy': 
            np.around(accuracy_score(y_test, y_pred_test), 3), 
        'balanced_accuracy': 
            np.around(balanced_accuracy_score(y_test, y_pred_test), 3),
        f"f1_{'binary' if y_train.nunique() == 2 else 'weighted'}": 
            np.around(f1_score(y_test, y_pred_test, average='binary' if y_train.nunique() == 2 else 'weighted', pos_label=pos_label), 3),
        "roc_auc":
            np.around(roc_auc_score(y_test, y_proba_test, multi_class='ovr'), 4)
    }

    st.write(pd.DataFrame([train_metrics, test_metrics], index=['train_set', 'test_set']))

    if hasattr(model, 'coef_'):
        st.markdown('### Coefficients')
        plot_data = model.coef_
        plot_xlabel = 'Coefficient'
    elif hasattr(model, 'feature_importances_'):
        st.markdown('### Feature Importances')
        plot_data = model.feature_importances_
        plot_xlabel = 'Feature importance'
    else:
        plot_data = None

    if (y_train.nunique() == 2 or algorithm == 'Decision Tree') and plot_data is not None:
        coef = pd.Series(
            data=np.squeeze(plot_data), 
            index=X_train.columns,
            name='coef'
        )
        coef = coef.rename_axis('column', axis=0)\
            .to_frame()\
            .reset_index()

        st.write(
            alt.Chart(coef).mark_bar().encode(
                x=alt.X('coef', title=plot_xlabel),
                y=alt.Y('column:O', title='Variable'),
                tooltip=alt.Tooltip('coef', format=".4f")
            ).properties(
                width=700,
                height=calc_height(X_train.shape[1])
            )
        )
    elif plot_data is not None:
        coef = pd.DataFrame(np.squeeze(plot_data), columns=X_train.columns).reset_index().melt(id_vars='index')
        maxV = np.ceil(np.max([np.abs(np.min(plot_data)), np.abs(np.max(plot_data))]))
        base = alt.Chart(coef).encode(
            x=alt.X('index:O', title='Class'),
            y=alt.Y('variable:O', title='Variable'),
        ).properties(
            width=700,
            height=calc_height(X_train.shape[1]) + 200
        )

        heatmap = base.mark_rect().encode(
            color=alt.Color('value', scale=alt.Scale(scheme='redblue', domain=[-maxV, 0, maxV]))
        )

        text = base.mark_text(baseline='middle').encode(
            text=alt.Text('value', format=".4f" if y_train.nunique() < 5 else ".2f")
        )

        st.write(
            heatmap + text
        )

    st.markdown('### Confusion Matrix')
    from sklearn.metrics import confusion_matrix

    def create_confusion(X, y, model, title):
        confusion = pd.DataFrame(confusion_matrix(y, model.predict(X), normalize='true'))
        confusion = confusion.reset_index().melt(id_vars='index')
        base = alt.Chart(confusion).encode(
            x=alt.X('index:O', title='Predicted Values'),
            y=alt.Y('variable:O', title='True Values'),
        ).properties(
            width=300,
            height=300,
            title=title
        )
        heatmap = base.mark_rect().encode(
            color=alt.Color('value', scale=alt.Scale(scheme='blues', domain=[0, 1]))
        )
        text = base.mark_text(baseline='middle').encode(
            text=alt.Text('value', format=".2f")
        )
        return (heatmap + text)

    st.write(
        alt.hconcat(
            create_confusion(X_train, y_train, model, 'Train set'),
            create_confusion(X_test, y_test, model, 'Test set')
        )
    )


    if y_train.nunique() == 2:
        st.markdown('### ROC Curve')
        from sklearn.metrics import roc_curve, auc

        def create_roc(X, y, model, pos_label, title):
            if algorithm in ['Logistic Regression']:
                decision = model.decision_function(X)
            if algorithm in ['Decision Tree', 'kNN', 'SVM']:
                decision = model.predict_proba(X)[:, 1]

            # Compute ROC curve and ROC area for each class
            fpr, tpr, _ = roc_curve(y, decision, pos_label=pos_label)
            roc_auc = auc(fpr, tpr)
            if roc_auc == 0.50:
                fpr, tpr, _ = roc_curve(y, decision, pos_label=0)
                roc_auc = auc(fpr, tpr)
            roc_df = pd.DataFrame([fpr, tpr]).T.rename(columns={0: 'fpr', 1: 'tpr'})
            base = alt.Chart(roc_df).mark_line().encode(
                x=alt.X('fpr', title='False Positive Rate'),
                y=alt.Y('tpr', title='True Positive Rate')
            ).properties(
                width=300,
                height=300,
                title=f"{title} (AUC: {np.around(roc_auc, 4)} )"
            )
            line = alt.Chart(
                pd.DataFrame({'x': [0, 1], 'y': [0, 1]})).mark_line(color='grey').encode(
                        alt.X('x'),
                        alt.Y('y'),
            )
            return (base + line)

        st.write(
            alt.hconcat(
                create_roc(X_train, y_train, model, pos_label, 'Train set'),
                create_roc(X_test, y_test, model, pos_label, 'Test set')
            )
        )

if problem_type == 'Regression':
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error

    train_metrics = {
        'r2_score': 
            np.around(r2_score(y_train, y_pred_train), 3), 
        'mean_absolute_error': 
            np.around(mean_absolute_error(y_train, y_pred_train), 3),
        'root_mean_squared_error': 
            np.around(np.sqrt(mean_squared_error(y_train, y_pred_train)), 3),
    }

    test_metrics = {
        'r2_score': 
            np.around(r2_score(y_test, y_pred_test), 3), 
        'mean_absolute_error': 
            np.around(mean_absolute_error(y_test, y_pred_test), 3),
        'root_mean_squared_error': 
            np.around(np.sqrt(mean_squared_error(y_test, y_pred_test)), 3),
    }

    st.write(pd.DataFrame([train_metrics, test_metrics], index=['train_set', 'test_set']))

    plot_coef = True
    if algorithm == 'SVM' and kernel != 'linear':
        plot_coef = False

    if algorithm not in ['kNN'] and plot_coef:
        if algorithm == 'Linear Regression':
            st.markdown('### Coefficients')
            plot_data = model.steps[1][1].coef_.tolist() + [model.steps[1][1].intercept_]
            plot_labels = model.steps[0][1].get_feature_names(X_train.columns) + ['intercept']
            plot_xlabel = 'Coefficient'
        else:
            if hasattr(model, 'coef_'):
                st.markdown('### Coefficients')
                if algorithm not in ['SVM']:
                    plot_data = model.coef_
                    if hasattr(model, 'intercept_'):
                        plot_data = plot_data + [model.intercept_]
                else:
                    plot_data = model.coef_.tolist()[0] + [model.intercept_[0]]
                plot_xlabel = 'Coefficient'
                plot_labels = X_train.columns.tolist() + ['intercept']
            if hasattr(model, 'feature_importances_'):
                st.markdown('### Feature Importances')
                plot_data = model.feature_importances_
                plot_xlabel = 'Feature importance'
                plot_labels = X_train.columns.tolist()
        
        coef = pd.Series(
            data=np.squeeze(plot_data), 
            index= plot_labels,
            name='coef'
        )
        coef = coef.rename_axis('column', axis=0)\
            .to_frame()\
            .reset_index()

        scale = 'symlog' if np.max(np.abs(coef['coef'])) - np.min(np.abs(coef['coef'])) > 1000 else 'linear'

        st.write(
            alt.Chart(coef).mark_bar().encode(
                x=alt.X('coef', title=plot_xlabel, scale=alt.Scale(type=scale)),
                y=alt.Y('column:O', title='Variable'),
                tooltip=[alt.Tooltip('coef', format=".4f"), alt.Tooltip('column')]
            ).properties(
                width=700,
                height=calc_height(len(plot_labels))
            )
        )

    st.markdown('### Diagnostic Plots')        

    plot_df = pd.concat((X_train, y_train), axis=1).reset_index()
    plot_df['fitted'] = y_pred_train
    plot_df['residual'] = plot_df['response'] - plot_df['fitted']

    residuals = alt.Chart(plot_df).mark_point().encode(
        x='index',
        y='residual'
    ).properties(
        width=650,
        height=200,
        title="Residuals"
    )

    plot_df = pd.concat((X_train, y_train), axis=1).reset_index()
    plot_df['fitted'] = y_pred_train

    fitted_vs_response = alt.Chart(plot_df).mark_point().encode(
        x='fitted',
        y=y_train.name
    ).properties(
        width=650,
        height=200,
        title="Fitted vs Response"
    )

    st.write(
        alt.vconcat(residuals, fitted_vs_response)
    )

    variable = st.selectbox('variable', X_train.columns)

    plot_df = pd.concat((X_train, y_train), axis=1).reset_index()
    plot_df['fitted'] = y_pred_train
    plot_df = plot_df.loc[:, ['index', variable, y_train.name, 'fitted']].melt(id_vars=['index', variable])

    one_vs_response = alt.Chart(plot_df).mark_point().encode(
        x=variable,
        y=alt.Y('value', title=y_train.name),
        color=alt.Color('variable', scale=alt.Scale(scheme='paired'))
    ).properties(
        width=700,
        height=400,
        title="One vs Response"
    )
    st.write(one_vs_response)
