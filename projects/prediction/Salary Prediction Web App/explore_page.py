import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

FILE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(FILE_DIR, "survey_results_public.csv")


def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = "Other"
    return categorical_map


def clean_experience(x):
    if x == "More than 50 years":
        return 50.0
    if x == "Less than 1 year":
        return 0.5
    try:
        return float(x)
    except (ValueError, TypeError):
        return 0.0


def clean_education(x):
    x = str(x)
    if "Bachelor" in x:
        return "Bachelor’s degree"
    elif "Master" in x:
        return "Master’s degree"
    elif "Professional" in x or "doctoral" in x or "Post" in x:
        return "Post grad"
    else:
        return "Less than a Bachelors"


@st.cache_data
def load_data():
    if not os.path.exists(CSV_PATH):
        return None

    df = pd.read_csv(CSV_PATH)
    needed_cols = ["Country", "EdLevel", "YearsCodePro", "Employment", "ConvertedCompYearly"]

    # Verify expected columns exist
    if not all(col in df.columns for col in needed_cols):
        return None

    df = df[needed_cols]
    df = df[df["ConvertedCompYearly"].notnull()]
    df = df[df["Employment"] == "Employed, full-time"]
    df = df.drop("Employment", axis=1)

    # Dynamic cutoff so both full and sample datasets work
    cutoff = 400 if len(df) > 5000 else 10
    country_map = shorten_categories(df.Country.value_counts(), cutoff)
    df["Country"] = df["Country"].map(country_map)
    df = df[df["ConvertedCompYearly"] <= 250000]
    df = df[df["ConvertedCompYearly"] >= 10000]
    df = df[df["Country"] != "Other"]

    df["YearsCodePro"] = df["YearsCodePro"].apply(clean_experience)
    df["EdLevel"] = df["EdLevel"].apply(clean_education)
    df = df.rename({"ConvertedCompYearly": "Salary"}, axis=1)
    return df


def show_explore_page():
    st.title("Explore Software Engineer Salaries")
    st.write("### Stack Overflow Developer Survey 2023")

    df = load_data()

    if df is None or df.empty:
        st.warning(
            "Dataset `survey_results_public.csv` was not found or is empty. "
            "Please run `python train.py` to generate sample data, or place the Stack Overflow survey dataset in this directory."
        )
        return

    data = df["Country"].value_counts()

    fig1, ax1 = plt.subplots()
    ax1.pie(
        data,
        labels=data.index,
        autopct="%1.1f%%",
        shadow=True,
        startangle=90,
        textprops={"fontsize": 6},
        labeldistance=1.1,
    )
    ax1.axis("equal")

    st.write("#### Number of Data from different countries")
    st.pyplot(fig1)
    plt.close(fig1)

    st.write("#### Mean Salary Based On Country")
    data_country = df.groupby(["Country"])["Salary"].mean().sort_values(ascending=True)
    st.bar_chart(data_country)

    st.write("#### Mean Salary Based On Experience")
    data_exp = df.groupby(["YearsCodePro"])["Salary"].mean().sort_values(ascending=True)
    st.line_chart(data_exp)


if __name__ == "__main__":
    show_explore_page()