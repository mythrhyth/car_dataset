import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from scipy.stats import skew, kurtosis
from scipy import stats


df = sns.load_dataset('car_crashes')

st.title('Data Exploration and Visualization')
st.write('This is a simple app to explore and visualize the car crashes dataset.')
st.write(f'Dataset shape: {df.shape[0]}rows and {df.shape[1]} columns')
st.write('Dataset columns:', df.columns.tolist())


st.sidebar.header('Filters')
feature = st.sidebar.selectbox('Select a feature to visualize', ['total', 'speeding', 'alcohol', 'not_distracted', 'no_previous', 'ins_premium'])
st.sidebar.subheader('Pairplot Features')
cols = ['total', 'speeding', 'alcohol', 'not_distracted', 'no_previous', 'ins_premium', 'ins_losses']
pair_cols = st.sidebar.multiselect("Select features for pairplot", cols, default=cols[:4])

if st.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.dataframe(df)

st.header("Univariate Analysis")
st.subheader("Summary Statistics")
cols = ['total', 'speeding', 'alcohol', 'not_distracted', 'no_previous', 'ins_premium', 'ins_losses']
desc_stats = df[cols].describe().T
desc_stats['skewness'] = df[cols].apply(skew)
desc_stats['kurtosis'] = df[cols].apply(kurtosis)
st.dataframe(desc_stats)


st.subheader(f"Distribution Plot for '{feature}'")
fig, ax = plt.subplots()
sns.histplot(df[feature], kde=True, ax=ax)
st.pyplot(fig)

st.subheader(f"Top 5 States by {feature.replace('_', ' ').title()}-Related Fatalities")
top5_alcohol = df[['abbrev', feature]].sort_values(by=feature, ascending=False).head(5)
fig = px.bar(top5_alcohol, x='abbrev', y=feature,  
             color_continuous_scale='Reds',
             color=feature,
             labels={'abbrev': 'State', 'alcohol': 'Alcohol Fatalities'})
st.plotly_chart(fig)

# 6. Q-Q Plot for Normality Check
st.subheader(f"Q-Q Plot for '{feature.replace('_', ' ').title()}'")

fig, ax = plt.subplots()
stats.probplot(df[feature], dist="norm", plot=ax)
st.pyplot(fig)

st.subheader(f"Box Plot for '{feature.replace('_', ' ').title()}'")
fig, ax = plt.subplots()
sns.boxplot(x=df[feature], ax=ax)
st.pyplot(fig)

st.header("Multivariate Analysis")

st.subheader("Correlation Heatmap")
corr = df[cols].corr()
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
st.pyplot(fig)


st.subheader("Pairplot of Selected Features")

if pair_cols:
    fig = sns.pairplot(df[pair_cols])
    st.pyplot(fig)
    