import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def append_columns_to_csv(file1, file2, output_file):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    if len(df1) != len(df2):
        raise ValueError("The two files must have the same number of rows")
    
    combined_df = pd.concat([df1, df2], axis=1)
    combined_df.to_csv(output_file, index=False)
    

def calculate_correlation(file1, col1, file2, col2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    if col1 not in df1.columns:
        raise ValueError(f"Column {col1} is not in {file1}")
    if col2 not in df2.columns:
        raise ValueError(f"Column {col2} is not in {file2}")
    
    if len(df1[col1]) != len(df2[col2]):
        raise ValueError("The columns from the two files must have the same number of rows")
    
    correlation = df1[col1].corr(df2[col2])
    return correlation

def sample_csv(input_csv, output_csv, sample_size=10000, random_state=42):

    df = pd.read_csv(input_csv)
    if len(df) < sample_size:
        raise ValueError(f"The input file has only {len(df)} rows, which is less than the sample size {sample_size}.")
    sampled_df = df.sample(n=sample_size, random_state=random_state)
    sampled_df.to_csv(output_csv, index=False)

def split_columns_to_csv(input_csv, col1, col2, output_csv1, output_csv2):
    df = pd.read_csv(input_csv)
    df1 = df[[col1]].rename(columns={col1: 'smiles'})
    df2 = df[[col2]].rename(columns={col2: 'smiles'})
    df1.to_csv(output_csv1, index=False)
    df2.to_csv(output_csv2, index=False)

def plot_correlation_and_save(file1, col1, file2, col2, output_file, name):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    if col1 not in df1.columns:
        raise ValueError(f"Column {col1} is not in {file1}")
    if col2 not in df2.columns:
        raise ValueError(f"Column {col2} is not in {file2}")
    
    if len(df1[col1]) != len(df2[col2]):
        raise ValueError("The columns from the two files must have the same number of rows")
    
    x = df1[col1]
    y = df2[col2]
    
    # Plot scatter
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c='skyblue', alpha=0.7, label=f"{name} vs {name}")
    
    z = np.polyfit(x, y, 1)  # Fit a linear model
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--", label="Trend line")
    
    plt.axhline(y.mean(), color='orange', linestyle='--', label=f"{name} Mean")
    plt.axvline(x.mean(), color='green', linestyle='--', label=f"{name} Mean")
    
    # Add titles and labels
    # plt.title(f"Scatter Plot of {col1} vs {col2}")
    plt.xlabel("Embedding Euclidean Distance")
    plt.ylabel(name)
    plt.legend()
    plt.grid(False)  # Remove grid
    plt.tight_layout()
    
    # Save plot to file
    plt.savefig(output_file)
    print(f"Scatter plot saved to {output_file}")
    plt.close()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

def plot_correlation_and_save_2(file1, col1, file2, col2, output_file, name):


    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    if col1 not in df1.columns:
        raise ValueError(f"Column {col1} is not in {file1}")
    if col2 not in df2.columns:
        raise ValueError(f"Column {col2} is not in {file2}")
    
    if len(df1[col1]) != len(df2[col2]):
        raise ValueError("The columns from the two files must have the same number of rows")
    
    x = df1[col1]
    y = df2[col2]
    
    # Calculate point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)  # KDE for density estimation
    idx = z.argsort()  # Sort points by density
    x, y, z = x[idx], y[idx], z[idx]
    
    # Plot scatter with density
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x, y, c=z, cmap='viridis', s=20, edgecolor='none', alpha=0.7, label=f"{name} vs {name}")
    plt.colorbar(scatter, label="Point Density")
    
    # Add trend line
    z_fit = np.polyfit(x, y, 1)  # Fit a linear model
    p = np.poly1d(z_fit)
    plt.plot(x, p(x), "r--", label="Trend line")
    
    # Add mean lines
    plt.axhline(y.mean(), color='orange', linestyle='--', label=f"{name} Mean")
    plt.axvline(x.mean(), color='green', linestyle='--', label=f"{name} Mean")
    
    # Add titles and labels
    plt.xlabel("Embedding Euclidean Distance")
    plt.ylabel(name)
    plt.legend()
    plt.grid(False)  # Remove grid
    plt.tight_layout()
    
    # Save plot to file
    plt.savefig(output_file)
    print(f"Scatter plot with density saved to {output_file}")
    plt.close()
    
def plot_correlation_and_save_3(file1, col1, file2, col2, output_file, name):
    plt.rcParams["font.size"] = 20
    # Read data
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    if col1 not in df1.columns:
        raise ValueError(f"Column {col1} is not in {file1}")
    if col2 not in df2.columns:
        raise ValueError(f"Column {col2} is not in {file2}")
    
    if len(df1[col1]) != len(df2[col2]):
        raise ValueError("The columns from the two files must have the same number of rows")
    
    x = df1[col1]
    y = df2[col2]
    
    # Calculate point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)  # Density estimation
    idx = z.argsort()  # Sort points by density
    x, y, z = x[idx], y[idx], z[idx]  # Sort x, y, z based on density

    # Plot scatter with density-based coloring
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x, y, c=z, cmap='viridis', s=20, alpha=0.7)
    # scatter = plt.scatter(x, y, c=z, cmap='viridis', s=20, alpha=0.7)
    plt.colorbar(scatter, label="Point Density")
    
    # Fit and plot trend line with varying width
    fit = np.polyfit(x, y, 1)
    trend = np.poly1d(fit)
    trend_x = np.linspace(x.min(), x.max(), 500)
    trend_y = trend(trend_x)
    
    # Calculate density along the trend line
    trend_xy = np.vstack([trend_x, trend_y])
    trend_density = gaussian_kde(xy)(trend_xy)
    normalized_density = (trend_density - trend_density.min()) / (trend_density.max() - trend_density.min())
    linewidths = 1 + 4 * normalized_density  # Linewidth varies between 1 and 5

    # Plot trend line with varying width
    for i in range(len(trend_x) - 1):
        plt.plot(
            trend_x[i:i+2], trend_y[i:i+2],
            color="green", alpha=0.8,
            linewidth=linewidths[i],
        )
    
    # Add mean lines
    plt.axhline(y.mean(), color='orange', linestyle='--', label=f"{name} Mean")
    plt.axvline(x.mean(), color='green', linestyle='--', label=f"Embedding Distance Mean")
    
    # Add labels and legend
    plt.xlabel("Embedding Euclidean Distance")
    plt.ylabel(name)
    plt.legend(fontsize=14)
    plt.grid(False)  # Remove grid
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file)
    print(f"Scatter plot with density and trend line saved to {output_file}")
    plt.close()
    