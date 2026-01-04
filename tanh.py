import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Page Configuration 
st.set_page_config(page_title="Tanh Activation Visualizer", layout="centered")

st.title("Hyperbolic Tangent (Tanh) Visualizer")
st.markdown("""
This app visualizes the **Tanh activation function**, which is commonly used in neural networks. 
It maps inputs to a value between **-1 and 1**.
""")

# Sidebar Controls
st.sidebar.header("Settings")

# Slider for input range (x-axis)
x_range = st.sidebar.slider(
    "Select Input Range (X-axis)",
    min_value=5,
    max_value=20,
    value=10,
    step=1,
    help="Adjusts the range of X from -value to +value"
)

# Slider for a specific point input
input_val = st.sidebar.slider(
    "Test a specific input value",
    min_value=-float(x_range),
    max_value=float(x_range),
    value=0.0,
    step=0.1
)

# Mathematical Logic 
# Generate X values
x = np.linspace(-x_range, x_range, 400)

# Tanh Formula: (e^x - e^-x) / (e^x + e^-x)
y = np.tanh(x)

# Calculate specific point
y_point = np.tanh(input_val)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the main Tanh curve
ax.plot(x, y, label="tanh(x)", color="#FF4B4B", linewidth=2.5)

# Plot the specific point selected by user
ax.scatter([input_val], [y_point], color="black", s=100, zorder=5, label=f"Input: {input_val}")
ax.vlines(input_val, -1.1, y_point, linestyles="dashed", colors="gray", alpha=0.5)
ax.hlines(y_point, -x_range, input_val, linestyles="dashed", colors="gray", alpha=0.5)

# Formatting the graph
ax.set_title("Hyperbolic Tangent Function", fontsize=16)
ax.set_xlabel("Input (x)", fontsize=12)
ax.set_ylabel("Output (tanh(x))", fontsize=12)
ax.set_ylim(-1.2, 1.2)
ax.set_xlim(-x_range, x_range)
ax.grid(True, linestyle="--", alpha=0.6)
ax.axhline(0, color='black', linewidth=1) # X-axis line
ax.axvline(0, color='black', linewidth=1) # Y-axis line
ax.legend()

# Display in Streamlit
st.pyplot(fig)

# Explanatory Metrics
col1, col2 = st.columns(2)

with col1:
    st.info(f"**Input (x):** {input_val}")

with col2:
    st.success(f"**Output (tanh):** {y_point:.4f}")

st.latex(r"tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}")

st.markdown("""
### Key Characteristics:
* **Zero-centered:** Unlike Sigmoid, the output is symmetric around 0.
* **Range:** $(-1, 1)$
* **Gradients:** Stronger gradients than Sigmoid (derivatives are steeper).
""")