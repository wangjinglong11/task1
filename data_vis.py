import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Set font to support English display
plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False  # Solve the problem of minus sign display


def load_data(file_path):
    """Load cancer dataset"""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully, {len(df)} rows and {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"Failed to load data: {e}")
        # Generate simulated data for demonstration
        print("Using simulated data for demonstration...")
        np.random.seed(42)
        df = pd.DataFrame({
            'TARGET_deathRate': np.random.normal(150, 30, 1000),
            'incidenceRate': np.random.normal(400, 80, 1000),
            'PctWhite': np.random.uniform(0, 100, 1000),
            'PctBlack': np.random.uniform(0, 100, 1000),
            'PctAsian': np.random.uniform(0, 100, 1000),
            'PctOtherRace': np.random.uniform(0, 100, 1000),
            'MedianAge': np.random.normal(40, 5, 1000),
            'povertyPercent': np.random.uniform(0, 30, 1000),
            'binnedInc': pd.cut(np.random.normal(50000, 15000, 1000), bins=5),
            'PctPrivateCoverage': np.random.uniform(0, 100, 1000),
            'PctPublicCoverage': np.random.uniform(0, 100, 1000),
        })
        return df


def plot_race_mortality(df, output_dir):
    """Plot bar chart of average cancer mortality rate by race"""
    race_columns = ['PctWhite', 'PctBlack', 'PctAsian', 'PctOtherRace']
    race_labels = ['White', 'Black', 'Asian', 'Other Races']
    race_data = []

    for col, label in zip(race_columns, race_labels):
        # Calculate average mortality rate for top 25% regions by race proportion
        high_race = df[df[col] > df[col].quantile(0.75)]
        avg_death_rate = high_race['TARGET_deathRate'].mean()
        race_data.append({'Race': label, 'AverageDeathRate': avg_death_rate})

    race_df = pd.DataFrame(race_data)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Race', y='AverageDeathRate', data=race_df, palette='viridis')

    # Add value labels on bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    fontsize=10, color='black',
                    xytext=(0, 5), textcoords='offset points')

    plt.title('Average Cancer Mortality Rate by Race Group')
    plt.xlabel('Race')
    plt.ylabel('Average Mortality Rate')
    plt.tight_layout()

    # Save plot to output directory
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/race_mortality.png', dpi=300)
    plt.show()


def plot_age_mortality(df, output_dir):
    """Plot bar chart of cancer mortality rate by age group"""
    # Create age groups
    df['AgeGroup'] = pd.cut(df['MedianAge'], bins=[0, 35, 45, 55, 100],
                            labels=['0-35', '36-45', '46-55', '56+'])

    # Calculate average mortality rate per age group
    age_grouped = df.groupby('AgeGroup')['TARGET_deathRate'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='AgeGroup', y='TARGET_deathRate', data=age_grouped, palette='magma')

    # Add value labels on bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    fontsize=10, color='black',
                    xytext=(0, 5), textcoords='offset points')

    plt.title('Average Cancer Mortality Rate by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Average Mortality Rate')
    plt.tight_layout()

    # Save plot to output directory
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/age_mortality.png', dpi=300)
    plt.show()


def plot_income_mortality(df, output_dir):
    """Plot bar chart of cancer mortality rate by income level"""
    # Calculate average mortality rate per income group
    income_grouped = df.groupby('binnedInc')['TARGET_deathRate'].mean().reset_index()

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='binnedInc', y='TARGET_deathRate', data=income_grouped, palette='plasma')

    # Add value labels on bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    fontsize=9, color='black',
                    xytext=(0, 5), textcoords='offset points')

    plt.title('Average Cancer Mortality Rate by Income Level')
    plt.xlabel('Income Group')
    plt.ylabel('Average Mortality Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save plot to output directory
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/income_mortality.png', dpi=300)
    plt.show()


def plot_insurance_mortality(df, output_dir):
    """Plot bar chart of cancer mortality rate by insurance coverage"""
    # Create insurance coverage groups
    df['HighPrivateCoverage'] = df['PctPrivateCoverage'] > df['PctPrivateCoverage'].median()
    df['HighPublicCoverage'] = df['PctPublicCoverage'] > df['PctPublicCoverage'].median()

    # Calculate average mortality rate for each combination
    insurance_data = []
    for private in [True, False]:
        for public in [True, False]:
            subset = df[(df['HighPrivateCoverage'] == private) & (df['HighPublicCoverage'] == public)]
            avg_death = subset['TARGET_deathRate'].mean()
            private_label = 'High Private Insurance' if private else 'Low Private Insurance'
            public_label = 'High Public Insurance' if public else 'Low Public Insurance'
            insurance_data.append({
                'Insurance Type': f'{private_label}\n{public_label}',
                'Average Mortality Rate': avg_death
            })

    insurance_df = pd.DataFrame(insurance_data)

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Insurance Type', y='Average Mortality Rate', data=insurance_df, palette='coolwarm')

    # Add value labels on bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    fontsize=10, color='black',
                    xytext=(0, 5), textcoords='offset points')

    plt.title('Average Cancer Mortality Rate by Insurance Coverage')
    plt.xlabel('Insurance Coverage')
    plt.ylabel('Average Mortality Rate')
    plt.tight_layout()

    # Save plot to output directory
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/insurance_mortality.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    # Load data
    df = load_data('../cancer_reg.csv')  # Replace with your actual data path

    # Create output directory
    output_directory = 'cancer_visualizations'

    # Plot mortality rate by race
    plot_race_mortality(df, output_directory)

    # Plot mortality rate by age group
    plot_age_mortality(df, output_directory)

    # Plot mortality rate by income level
    plot_income_mortality(df, output_directory)

    # Plot mortality rate by insurance coverage
    plot_insurance_mortality(df, output_directory)