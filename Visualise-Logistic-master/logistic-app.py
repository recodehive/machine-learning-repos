import streamlit as st
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

from sklearn.datasets import make_classification


def main():
    st.title("LOGISTIC REGRESSION MODEL")
    st.sidebar.title("Logistic Regression Web App")

    @st.cache(persist=True)
    def load_data(random_var):
        X1, Y1 = make_classification(n_samples=100, n_features=2, n_redundant=0,
                                     n_informative=1, n_clusters_per_class=1, random_state=random_var)
        col_x = []
        col_y = []
        for i in range(len(X1)):
            col_x.append(X1[i][0])
            col_y.append(X1[i][1])
        data = pd.DataFrame({' X ': col_x, ' Y ': col_y, 'Classification': Y1})
        return data

    # @st.cache(persist=True)
    def split(df):
        y = df.Classification.values
        x = df.drop(columns=['Classification']).values
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list, model, trained_model, x_train, y_train, r):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test, y_test,
                                  display_labels=class_names)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()

        if 'Decision Boundary' in metrics_list:
            st.write("Decision Boundary")
            d_f = load_data(r)

            xaxis = np.arange(start=x_train[:, 0].min(
            ) - 1, stop=x_train[:, 0].max() + 1, step=0.01)
            yaxis = np.arange(start=x_train[:, 1].min(
            ) - 1, stop=x_train[:, 1].max() + 1, step=0.01)
            xx, yy = np.meshgrid(xaxis, yaxis)

            in_array = np.array([xx.ravel(), yy.ravel()]).T
            labels = trained_model.predict(in_array)

            plt.contourf(xx, yy, labels.reshape(
                xx.shape), alpha=0.5, cmap="RdBu")
            plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap="RdBu", vmin=-.2, vmax=1.2,
                        edgecolor="white", linewidth=1)

            st.pyplot()

    r = 4

    if st.sidebar.checkbox("Show Raw Data", True):
        random_data = st.sidebar.selectbox(
            "Select Data", (1, 2, 3, 4, 5), index=0)

        #R = np.random.RandomState()
        #r = R.randint(5, 15)
        # df = load_data(None)
        # class_names = ['0', '1']
        # x_train, x_test, y_train, y_test = split(df)
        df = load_data(random_data)
        data_table = st.empty()
        data_table.dataframe(df)

    if st.sidebar.checkbox("Logistic Regression", False):
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input(
            "C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider(
            "Maximum number of iterations", 100, 500, key='max_iter')

        st.sidebar.subheader("Choose Solver ")
        solverr = st.sidebar.selectbox(
            "Solver", ("liblinear", "newton-cg", "lbfgs", "sag", "saga"))

        metrics = st.sidebar.multiselect("What metrics to plot?", (
            'Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve', 'Decision Boundary'))

        class_names = ['0', '1']

        x_train, x_test, y_train, y_test = split(df)

        # @st.cache(persist=True)
        def common(s, p):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(
                C=C, penalty=p, max_iter=max_iter, solver=s)
            trained_model = model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(
                y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(
                y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics, model, trained_model,
                         x_train, y_train, r)

        if (solverr == 'newton-cg'):
            st.subheader(
                "For multiclass problems, ‘newton-cg’handle multinomial loss .‘newton-cg’ handle only l2 penalty or no")
            if st.sidebar.button("Classify", key='c_1'):
                common('newton-cg', 'l2')

        if (solverr == 'lbfgs'):
            st.subheader(
                "For multiclass problems, ‘lbfgs’ handle multinomial loss .‘lbfgs’ handle only l2 penalty or no")
            if st.sidebar.button("Classify", key='c_2'):
                common('lbfgs', 'l2')

        if (solverr == 'sag'):
            st.subheader(
                "For multiclass problems, ‘sag’ handle multinomial loss .‘sag’ handle only l2 penalty or no")
            if st.sidebar.button("Classify", key='c_3'):
                common('sag', 'l2')

        if solverr == 'liblinear':
            st.subheader(
                " For small datasets, ‘liblinear’ is a good choice and the default solver. It is limited to one-versus-rest schemes. It handle l1 penalty")
            if st.sidebar.button("Classify", key='c_4'):
                common('liblinear', 'l2')

        if solverr == 'saga':
            st.subheader(
                "For multiclass problems, ‘saga’ handle multinomial loss .‘saga’ handle l1, l2, elastic penalty . ‘saga’ are faster for large datasets. ")
            penaltyy = st.sidebar.radio(
                "Choose Penalty", ('l1', 'l2', 'elasticnet'), key='pe')

            if penaltyy == 'l1':
                st.markdown("For l1 Penalty")
                if st.sidebar.button("Classify", key='c_5'):
                    common('saga', 'l1')

            if penaltyy == 'l2':
                st.markdown("For l2 Penalty")
                if st.sidebar.button("Classify", key='c_6'):
                    common('saga', 'l2')

            if penaltyy == 'elasticnet':
                st.markdown("For elasticnet Penalty")
                if st.sidebar.button("Classify", key='c_7'):
                    common('saga', 'elasticnet')


if __name__ == '__main__':
    main()
