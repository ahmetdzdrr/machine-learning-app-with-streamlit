import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def data_analysis(dataframe):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        checkbox1 = st.checkbox("Shape")

    with col2:
        checkbox2 = st.checkbox("Duplicate Value")

    with col3:
        checkbox3 = st.checkbox("Null Values")
    with col4:
        checkbox4 = st.checkbox("Columns")

    col_first, col_second = st.columns(2)

    with (col_first):
        if checkbox1:
            st.write("Dataframe Shape")
            st.success(dataframe.shape, icon="✅")
        if checkbox2:
            st.write("Dataframe Duplicate Values")
            st.success(dataframe.duplicated().sum(), icon="✅")
        if checkbox3:
            st.write("Dataframe Null Values")
            if dataframe.isnull().any().any():
                for col in dataframe.columns:
                    if dataframe[col].isnull().sum() > 0:
                        st.success(f"{col}: {dataframe[col].isnull().sum()}", icon="✅")
            else:
                st.success(dataframe.isnull().sum().sum())

    with (col_second):
        if checkbox4:
            st.write("Data Columns")
            st.table(dataframe.columns)

    st.markdown("---")


def data_preprocessing(dataframe):
    st.write("### Data Preprocessing")
    options = st.multiselect("Choose columns for removing",
                             dataframe.columns, default=None)

    if len(options) > 0:
        dataframe.drop(options, axis=1, inplace=True)
        st.write([dataframe.columns])
        st.success("The selected columns have been deleted", icon="✅")

    else:
        st.warning("You must select at least a column", icon="⚠️")


def processing_null_values(dataframe):
    st.write("### Check Values")

    col1, col2 = st.columns(2)
    with col1:
        checkbox1 = st.checkbox("Remove Null Values")
    with col2:
        checkbox2 = st.checkbox("Fill Null Values")

    if dataframe.isnull().any().any():
        if checkbox1:
            dataframe.dropna(inplace=True)
            st.success("All null values removed", icon="✅")
        else:
            if checkbox2:
                num_cols = dataframe.select_dtypes(include=[int, float]).columns.tolist()
                object_cols = dataframe.select_dtypes(include=["object", "category"]).columns.tolist()
                option = st.selectbox("Choose your dataframe target", ["None"] + num_cols + object_cols)

                if option != 'None':
                    target = option
                    for col in num_cols:
                        dataframe[col] = dataframe.groupby(target)[col].transform(
                            lambda x: x.fillna(x.median()))
                    else:
                        dataframe[col] = dataframe.groupby(target)[col].transform(
                            lambda x: x.fillna(x.mode().iloc[0]))
                    st.success("Dataframe null values filled", icon="✅")
                else:
                    st.info("Please select target column", icon="ℹ️")

                    if dataframe.isnull().any().any():
                        st.warning("Dataframe has null values", icon="⚠️")
                    else:
                        st.info("Dataframe has not null values. Please pass this section", icon="ℹ️")
    else:
        st.info("Dataframe has not any null values", icon="ℹ️")


def encoding_process(dataframe):
    col1, col2, col3 = st.columns(3)

    with col1:
        checkbox1 = st.checkbox("Label Encoder")
    with col2:
        checkbox2 = st.checkbox("Ordinal Encoder")
    with col3:
        checkbox3 = st.checkbox("OneHot Encoder")

    object_cols = dataframe.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if len(object_cols) > 0:
        if checkbox1 or checkbox2 or checkbox3:
            pass
        else:
            st.warning("You must choose at least an encoding process", icon="⚠️")

        if checkbox1:
            label_encoder = LabelEncoder()
            for column in dataframe.columns:
                if ((dataframe[column].dtype in ["object", "category", "bool"]) and \
                        (dataframe[column].nunique() == 2)):
                    dataframe[column] = label_encoder.fit_transform(dataframe[column])

        if checkbox1 and checkbox2:
            ordinal_encoder = OrdinalEncoder()
            for column in dataframe.columns:
                if (dataframe[column].dtype in ["object", "category", "bool"]) and \
                        (dataframe[column].nunique() > 2):
                    dataframe[column] = ordinal_encoder.fit_transform(dataframe[column].values.reshape(-1, 1))
            st.write(dataframe.head())
            st.success("Label and Ordinal encoding processes are done", icon="✅")
            return dataframe

        if checkbox3:
            categorical_columns = dataframe.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
            if len(categorical_columns) > 0:
                dataframe = pd.get_dummies(dataframe, columns=categorical_columns, drop_first=True)
                st.write(dataframe.head())
                st.success("OneHot encoding process is done", icon="✅")
            else:
                st.info("You don't need to choose OneHot Encoder option", icon="ℹ️")

        if checkbox1:
            if len(object_cols) > 0:
                st.warning("You have still object or category columns. "
                           "Please choose one more encoding process", icon="⚠️")
            else:
                st.success("Label encoding process is done", icon="✅")

    else:
        st.info("Dataframe columns have not any object, category or bool data types", icon="ℹ️")



def scaling_process(dataframe):
    options = st.selectbox("Choose an encoding process", ["None", "Standard Scaler",
                                                          "Robust Scaler", "MinMax Scaler"])

    if options == "Standard Scaler":
        ss = StandardScaler()
        cols = [col for col in dataframe.columns if dataframe[col].nunique() > 10]
        if len(cols) > 0:
            dataframe[cols] = ss.fit_transform(dataframe[cols])
            st.write("Scaled first 5 observations:")
            st.write(dataframe.head())
        st.success("Standard scaling process is done", icon="✅")
        return dataframe

    elif options == "Robust Scaler":
        rs = RobustScaler()
        cols = [col for col in dataframe.columns if dataframe[col].nunique() > 10]
        if len(cols) > 0:
            dataframe[cols] = rs.fit_transform(dataframe[cols])
            st.write("Scaled first 5 observations:")
            st.write(dataframe.head())
        st.success("Robust scaling process is done", icon="✅")
        return dataframe

    elif options == "MinMax Scaler":
        mms = MinMaxScaler()
        cols = [col for col in dataframe.columns if dataframe[col].nunique() > 10]
        if len(cols) > 0:
            dataframe[cols] = mms.fit_transform(dataframe[cols])
            st.write("Scaled first 5 observations:")
            st.write(dataframe.head())
        st.success("MinMax scaling process is done", icon="✅")
        return dataframe

    else:
        st.warning("You must select an encoding process", icon="⚠️")

def select_model(dataframe):
    columns = dataframe.columns.tolist()
    feature_target = st.selectbox("Choose target variable", ["None"] + columns)

    if feature_target != 'None':
        X = dataframe.drop(feature_target, axis=1)
        y = dataframe[feature_target]
        col_x = X.columns.tolist()
        col_y = y.name
        col1, col2 = st.columns(2)
        with col1:
            st.write("X variables:", col_x)
        with col2:
            st.write("y variable:", [col_y])
    else:
        st.warning("Please choose your target variable", icon="⚠️")

    train_size_option = st.selectbox("Train size", ["None"] + [0.66, 0.7, 0.75, 0.80])
    random_state_option = st.slider("Random state", 0, 100, 1)
    model_options = st.selectbox(
        "Choose a model",
        ["None", "Logistic Regression", "Decision Tree", "SVM",
         "Random Forest", "XGBoost", "GradientBoosting"]
    )
    submit_model = st.button("Apply", key="submit_button")

    if submit_model:
        if (feature_target != 'None') and (train_size_option != 'None'):
            X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                                train_size=train_size_option, \
                                                                test_size=1 - train_size_option, \
                                                                random_state=random_state_option)

            if model_options == 'None':
                st.error("You must choose a model", icon="🚨")
            else:
                if model_options == 'Logistic Regression':
                    lr = LogisticRegression()
                    lr.fit(X_train, y_train)
                    y_pred = lr.predict(X_test)

                elif model_options == 'SVM':
                    svm = SVC()
                    svm.fit(X_train, y_train)
                    y_pred = svm.predict(X_test)

                elif model_options == 'Random Forest':
                    rf = RandomForestClassifier()
                    rf.fit(X_train, y_train)
                    y_pred = rf.predict(X_test)

                elif model_options == 'XGBoost':
                    xgb = XGBClassifier()
                    xgb.fit(X_train, y_train)
                    y_pred = xgb.predict(X_test)

                elif model_options == 'Decision Tree':
                    d_tree = tree.DecisionTreeClassifier()
                    d_tree.fit(X_train, y_train)
                    y_pred = d_tree.predict(X_test)

                elif model_options == 'GradientBoosting':
                    gboost = GradientBoostingClassifier()
                    gboost.fit(X_train, y_train)
                    y_pred = gboost.predict(X_test)

                st.success(f"You choosed  {model_options}", icon="✅")
                accuracy = accuracy_score(y_test, y_pred)
                st.success(f"Accuracy: {accuracy:.4f}")

                st.write("### Classification Report:")
                report = classification_report(y_test, y_pred, target_names=['0', '1'], output_dict=True)
                report_df = pd.DataFrame(report).transpose()

                def highlight(s):
                    is_max = s == s.max()
                    return ['background-color: gray' if v else '' for v in is_max]

                report_df.style.apply(highlight, axis=0)
                st.dataframe(report_df.style.apply(highlight, axis=0))

                st.write("### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=0.5, square=True)
                plt.title(f"{model_options}")
                plt.xlabel("Predicted Labels")
                plt.ylabel("True Labels")
                st.pyplot()


        else:
            st.error("Please choose other options", icon="🚨")


def pie_visualization(dataframe):
    columns = [col for col in dataframe.columns if dataframe[col].nunique() < 6]
    column_option = st.selectbox("Choose a feature for Pie Chart:", ['None'] + columns)
    if column_option != 'None':
        target_counts = dataframe[column_option].value_counts()

        colors = [plt.cm.Paired(random.choice(range(12))) for _ in range(len(target_counts))]

        wedgeprops = {'linewidth': 2, 'edgecolor': 'white'}
        fig, ax = plt.subplots()
        ax.pie(target_counts, labels=target_counts.index, autopct='%1.2f%%',
               startangle=90, wedgeprops=wedgeprops, colors=colors)
        ax.axis('equal')
        ax.set_title(f"{column_option} Pie Chart")
        st.pyplot(fig)


def barplot_visualization(dataframe):
    columns = [col for col in dataframe.columns if dataframe[col].nunique() > 6]
    columns_option = st.selectbox("Choose a feature for Bar Chart:", ["None"] + columns)

    if columns_option != 'None':
        plt.figure(figsize=(12, 6))
        plt.title(f"{columns_option} Bar Plot")
        sns.set(font_scale=1.5)
        sns.countplot(data=dataframe, x=columns_option)
        st.pyplot()


def feature_comparison(dataframe):
    select_first_feature = st.multiselect("Choose first feature:", dataframe.columns, default=None)
    select_second_feature = st.multiselect("Choose second feature:", dataframe.columns, default=None)

    if select_first_feature and select_second_feature:
        for column in select_first_feature:
            for second_column in select_second_feature:
                if column != second_column:
                    plt.figure(figsize=(8, 6))
                    sns.countplot(x=column, hue=second_column, data=dataframe, palette="hls")
                    plt.title(f"{column} vs {second_column}")
                    st.pyplot()


def correlation_matrix(dataframe):
    features = dataframe.select_dtypes(include=[int, float])
    correlation = features.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation, annot=True, fmt="0.3f", cmap="coolwarm")
    st.pyplot()
