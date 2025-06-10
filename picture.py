import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
            'MedianAgeMale': np.random.normal(40, 5, 1000),
            'MedianAgeFemale': np.random.normal(42, 5, 1000),
            'povertyPercent': np.random.uniform(0, 30, 1000),
            'binnedInc': pd.cut(np.random.normal(50000, 15000, 1000), bins=5)
        })
        return df


def plot_race_cancer_distribution(df, cancer_metric='TARGET_deathRate'):
    """Plot boxplot of cancer metrics distribution by race"""
    race_columns = ['PctWhite', 'PctBlack', 'PctAsian', 'PctOtherRace']

    # Create a dataframe where each row represents the proportion of a race in a region
    # and the corresponding cancer metric
    race_data = []
    for race_col in race_columns:
        race_name = race_col.replace('Pct', '')
        # Divide into high and low proportion groups based on the median
        high_race = df[df[race_col] > df[race_col].median()]
        low_race = df[df[race_col] <= df[race_col].median()]

        race_data.append({
            'Race': race_name,
            'Group': 'High',
            cancer_metric: high_race[cancer_metric].values
        })
        race_data.append({
            'Race': race_name,
            'Group': 'Low',
            cancer_metric: low_race[cancer_metric].values
        })

    # Convert to long-format dataframe
    race_df = pd.DataFrame(race_data)

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Race', y=cancer_metric, hue='Group', data=race_df.explode(cancer_metric))
    plt.title(f'Comparison of {cancer_metric} by Race Distribution')
    plt.xlabel('Race')
    plt.ylabel(cancer_metric)
    plt.legend(title='Proportion Group')
    plt.tight_layout()
    plt.savefig(f'race_{cancer_metric}_distribution.png', dpi=300)
    plt.show()


def plot_gender_age_cancer(df, cancer_metric='incidenceRate'):
    """Plot boxplot of cancer metrics distribution by gender and age"""
    # Create age groups
    df['AgeGroup'] = pd.cut(df['MedianAge'], bins=5)

    plt.figure(figsize=(14, 7))

    # Relationship between age and cancer metrics
    plt.subplot(1, 2, 1)
    sns.boxplot(x='AgeGroup', y=cancer_metric, data=df)
    plt.title(f'Distribution of {cancer_metric} by Age Group (Overall)')
    plt.xlabel('Age Group')
    plt.ylabel(cancer_metric)
    plt.xticks(rotation=30)

    # Save the image
    plt.tight_layout()
    plt.savefig(f'age_{cancer_metric}_distribution.png', dpi=300)
    plt.show()


def plot_socioeconomic_cancer(df, cancer_metric='TARGET_deathRate'):
    """Plot boxplot of the relationship between socioeconomic factors and cancer metrics"""
    plt.figure(figsize=(16, 8))

    # Relationship between poverty rate and cancer metrics
    plt.subplot(1, 2, 1)
    df['PovertyGroup'] = pd.qcut(df['povertyPercent'], 4)
    sns.boxplot(x='PovertyGroup', y=cancer_metric, data=df)
    plt.title(f'Distribution of {cancer_metric} by Poverty Rate Group')
    plt.xlabel('Poverty Rate Group')
    plt.ylabel(cancer_metric)
    plt.xticks(rotation=15)

    # Relationship between income level and cancer metrics
    plt.subplot(1, 2, 2)
    sns.boxplot(x='binnedInc', y=cancer_metric, data=df)
    plt.title(f'Distribution of {cancer_metric} by Income Level Group')
    plt.xlabel('Income Group')
    plt.ylabel(cancer_metric)
    plt.xticks(rotation=30)

    plt.tight_layout()
    plt.savefig(f'socioeconomic_{cancer_metric}_distribution.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    # Load data
    df = load_data('cancer_reg.csv')  # Replace with your actual data path

    # Plot cancer death rate distribution by race
    plot_race_cancer_distribution(df, 'TARGET_deathRate')

    # Plot cancer incidence rate distribution by gender and age
    plot_gender_age_cancer(df, 'incidenceRate')

    # Plot the relationship between socioeconomic factors and cancer death rate
    plot_socioeconomic_cancer(df, 'TARGET_deathRate')