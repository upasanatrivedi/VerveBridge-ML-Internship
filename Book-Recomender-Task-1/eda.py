import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path):
    logger.info("Loading data from %s", file_path)
    df = pd.read_csv(file_path)
    logger.info("Data loaded successfully")
    return df

def explore_dataset(df):
    logger.info("Exploring dataset")
    print("Dataset Information:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe())
    logger.info("Dataset exploration complete")

def plot_distribution(df):
    try:
        logger.info("Plotting distributions")

        plt.figure(figsize=(10, 8))
        sns.countplot(x='Height', data=df)
        plt.title('Height Distribution')
        plt.xlabel('Height')
        plt.ylabel('Count')
        plt.show()

        plt.figure(figsize=(10, 8))
        sns.countplot(y='Genre', data=df, order=df['Genre'].value_counts().index)
        plt.title('Genres Distribution')
        plt.xlabel('Count')
        plt.ylabel('Genre')
        plt.show()

        plt.figure(figsize=(10, 8))
        sns.countplot(y='Author', data=df, order=df['Author'].value_counts().head().index)
        plt.title('Top 10 Authors by Book Count')
        plt.xlabel('Count')
        plt.ylabel('Author')
        plt.show()

        plt.figure(figsize=(10, 8))
        sns.countplot(y='Publisher', data=df, order=df['Publisher'].value_counts().head().index)
        plt.title('Top 10 Publishers by Book Count')
        plt.xlabel('Count')
        plt.ylabel('Publisher')
        plt.show()

    except Exception as e:
        logger.error("Error while plotting distributions: %s", e)

def plot_top_n(df, column, n, title, xlabel, ylabel):
    logger.info("Plotting top %d values of %s", n, column)
    plt.figure(figsize=(8, 6))
    sns.countplot(y=column, data=df, order=df[column].value_counts().iloc[:n].index)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    logger.info("Plotting complete")

def main():
    file_path = "books.csv"
    df = load_data(file_path)
    explore_dataset(df)

    plot_distribution(df, "Height", "Height Distribution", "Height", "Count")
    plot_distribution(df, "Genre", "Genre Distribution", "Genre", "Count")

    plot_top_n(df, "Author", 10, "Top 10 Authors by Book Count", "Count", "Author")
    plot_top_n(df, "Publisher", 10, "Top 10 Publishers by Book Count", "Count", "Publisher")

if __name__ == "__main__":
    main()