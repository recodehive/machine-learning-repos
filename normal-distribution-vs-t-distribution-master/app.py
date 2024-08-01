import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t

st.set_page_config(page_title='Normal and t-Distribution Comparison', layout='wide')

st.title('Normal and t-Distribution Comparison')


# Input values
with st.sidebar:
    st.header('Input Parameters')
    degrees_of_freedom = st.slider('Degrees of Freedom', 1, 100, 5, 1)

# Generate data for normal distribution
x = np.linspace(-5, 5, 1000)
y_norm = norm.pdf(x)

# Generate data for t-distribution
y_t = t.pdf(x, df=degrees_of_freedom)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x, y_norm, label='Normal Distribution', color='blue')
ax.plot(x, y_t, label=f"Student's t-Distribution (df={degrees_of_freedom})", color='red')
ax.set_xlabel('x')
ax.set_ylabel('Probability Density')
ax.set_title('Normal and t-Distribution Comparison')
ax.legend()

# Display the plot
st.pyplot(fig)
