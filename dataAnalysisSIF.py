"""
Exploratory Data Analysis (EDA) Script for Parquet Data
MVP implementation with pandas, matplotlib, and seaborn
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_data(file_path="data.parquet"):
    """
    Load parquet file into pandas DataFrame
    
    Args:
        file_path (str): Path to the parquet file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    print("=" * 50)
    print("LOADING DATA")
    print("=" * 50)
    
    try:
        df = pd.read_parquet(file_path)
        print(f"✓ Successfully loaded data from {file_path}")
        return df
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None

def profile_data(df):
    """
    Perform basic data profiling and statistics
    
    Args:
        df (pd.DataFrame): Input DataFrame
    """
    print("\n" + "=" * 50)
    print("DATA PROFILING")
    print("=" * 50)
    
    # Basic dataset information
    print(f"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Column names and types
    print(f"\nColumn Names ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col} ({df[col].dtype})")
    
    # First 5 rows
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    # Basic info
    print(f"\nDataFrame Info:")
    df.info()
    
    # Statistical summary
    print(f"\nStatistical Summary (include='all'):")
    print(df.describe(include='all'))
    
    # Missing values
    print(f"\nMissing Values:")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing_Count': missing_data.values,
        'Missing_Percentage': missing_percent.values
    }).sort_values('Missing_Count', ascending=False)
    
    if missing_df['Missing_Count'].sum() == 0:
        print("✓ No missing values found!")
    else:
        print(missing_df[missing_df['Missing_Count'] > 0])
    
    # Categorical columns analysis
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print(f"\nCategorical Columns Analysis:")
        for col in categorical_cols:
            print(f"\n{col} - Top 10 Value Counts:")
            print(df[col].value_counts().head(10))
            print(f"Unique values: {df[col].nunique()}")
    else:
        print(f"\n✓ No categorical columns found")

def visualize_data(df):
    """
    Create visualizations for data exploration
    
    Args:
        df (pd.DataFrame): Input DataFrame
    """
    print("\n" + "=" * 50)
    print("DATA VISUALIZATION")
    print("=" * 50)
    
    # Set style for better looking plots
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    if len(numeric_cols) == 0:
        print("✓ No numeric columns found for visualization")
        return
    
    # Create histograms for numeric columns
    print(f"Creating histograms for {len(numeric_cols)} numeric columns...")
    
    # Filter out columns with problematic data
    valid_numeric_cols = []
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            print(f"  ⚠ Skipping {col}: all values are NaN")
        elif col_data.isin([float('inf'), float('-inf')]).any():
            print(f"  ⚠ Skipping {col}: contains infinite values")
        elif col_data.min() == col_data.max():
            print(f"  ⚠ Skipping {col}: constant values (min=max={col_data.min()})")
        else:
            valid_numeric_cols.append(col)
    
    if len(valid_numeric_cols) == 0:
        print("✓ No valid numeric columns for histogram visualization")
        return
    
    # Calculate subplot grid with better sizing
    n_cols = min(2, len(valid_numeric_cols))  # Reduce columns for better readability
    n_rows = (len(valid_numeric_cols) + n_cols - 1) // n_cols
    
    # Dynamic figure sizing based on number of plots
    fig_width = max(12, n_cols * 6)
    fig_height = max(8, n_rows * 4)
    plt.figure(figsize=(fig_width, fig_height))
    
    for i, col in enumerate(valid_numeric_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        
        # Clean data for histogram
        clean_data = df[col].dropna()
        
        # Handle edge cases with data range
        data_min, data_max = clean_data.min(), clean_data.max()
        
        # Create histogram with safe parameters
        try:
            clean_data.hist(bins=30, alpha=0.7, edgecolor='black')
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
        except ValueError as e:
            print(f"  ⚠ Could not create histogram for {col}: {e}")
            plt.text(0.5, 0.5, f'Cannot visualize\n{col}\nRange: {data_min:.2e} to {data_max:.2e}', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f'Distribution of {col} (Visualization Failed)')
    
    # Improve layout spacing
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Add more spacing between subplots
    plt.show()
    
    # Correlation heatmap for numeric columns
    if len(valid_numeric_cols) > 1:
        print("Creating correlation heatmap...")
        
        # Dynamic heatmap sizing based on number of variables
        n_vars = len(valid_numeric_cols)
        fig_size = max(8, min(12, n_vars * 1.5))  # Scale with number of variables
        
        plt.figure(figsize=(fig_size, fig_size))
        correlation_matrix = df[valid_numeric_cols].corr()
        
        # Create heatmap with better formatting
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={'shrink': 0.8},
                   annot_kws={'size': max(8, 14 - n_vars)})  # Adjust font size based on variables
        
        plt.title('Correlation Heatmap of Numeric Variables', fontsize=16, pad=20)
        plt.tight_layout(pad=2.0)
        plt.show()
        
        # Print correlation insights
        print("\nCorrelation Insights:")
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        corr_val
                    ))
        
        if high_corr_pairs:
            print("High correlations (|r| > 0.7):")
            for col1, col2, corr in high_corr_pairs:
                print(f"  {col1} ↔ {col2}: {corr:.3f}")
        else:
            print("✓ No high correlations found between numeric variables")
    else:
        print("✓ Need at least 2 numeric columns for correlation analysis")

def main():
    """
    Main function to orchestrate the EDA workflow
    """
    print("EXPLORATORY DATA ANALYSIS (EDA) - MVP")
    print("=" * 60)
    
    # Step 1: Load data
    df = load_data("data.parquet")
    if df is None:
        print("Failed to load data. Exiting...")
        return
    
    # Step 2: Profile data
    profile_data(df)
    
    # Step 3: Visualize data
    visualize_data(df)
    
    print("\n" + "=" * 60)
    print("EDA COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Summary:")
    print(f"• Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"• Numeric columns: {len(df.select_dtypes(include=['number']).columns)}")
    print(f"• Categorical columns: {len(df.select_dtypes(include=['object', 'category']).columns)}")
    print(f"• Missing values: {df.isnull().sum().sum()} total")

if __name__ == "__main__":
    main()