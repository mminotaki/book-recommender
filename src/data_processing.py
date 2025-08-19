import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def heatmap_missing_values(df, figsize=(12, 6)):
    """
    Plots a heatmap of missing values for a given DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset to check for missing values.
    figsize : tuple, optional (default=(12, 6))
        Size of the matplotlib figure.
    """
    plt.figure(figsize=figsize)
    ax = plt.axes()
    sns.heatmap(df.isna().transpose(), 
                cbar=False, 
                ax=ax, 
                cmap="binary")

    ax.set_xlabel("Columns")
    ax.set_ylabel("Missing Values")
    plt.show()



 
def plot_correlation_matrix(df, columns_of_interest, method='spearman', figsize=(8,6), cmap='coolwarm'):
    """
    Plots a correlation heatmap for selected columns of a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset containing the columns to analyze.
    columns_of_interest : list of str
        List of column names to include in the correlation matrix.
    method : str, optional (default='spearman')
        Correlation method: 'pearson', 'spearman', or 'kendall'.
    figsize : tuple, optional (default=(8,6))
        Size of the figure.
    cmap : str, optional (default='coolwarm')
        Colormap for the heatmap.
    """
    # Compute correlation matrix using the specified method
    correlation_matrix = df[columns_of_interest].corr(method=method)

    sns.set_theme(style='white')
    plt.figure(figsize=figsize)
    heatmap = sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap=cmap,
                          cbar_kws={'label': f'{method.capitalize()} correlation'})
    
    plt.show()



def plot_category_distribution(df, category_column='categories', title='Category Distribution in Dataset'):
    """
    Plots a bar chart showing the distribution of categories in a dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset containing the category column.
    category_column : str, optional (default='categories')
        The column name containing categorical values.
    title : str, optional (default='Category Distribution in Dataset')
        Title of the bar chart.
    """
    # Prepare the data
    category_counts = df[category_column].value_counts().reset_index()
    category_counts.columns = ['category', 'count']

    # Sort by count (most popular categories first)
    category_counts = category_counts.sort_values(by='count', ascending=False)

    # Create the bar chart
    fig = px.bar(
        category_counts,
        x='category',
        y='count',
        title=title,
        labels={'category': 'Category', 'count': 'Count'},
        # color='count',  # optional: adds a color gradient
        # color_continuous_scale='Blues'
    )

    # Customize layout
    fig.update_layout(
        xaxis_tickangle=90,
        xaxis_title='Category',
        yaxis_title='Count',
        template='plotly_white'
    )

    # Show the plot
    fig.show()



def plot_description_length_distribution(df, column='words_in_description', nbins=100, title='Distribution of Words in Description'):
    """
    Plots a histogram showing the distribution of description lengths in a dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset containing the description length column.
    column : str, optional (default='words_in_description')
        The column containing the number of words or length of descriptions.
    nbins : int, optional (default=100)
        Number of bins for the histogram.
    title : str, optional (default='Distribution of Words in Description')
        Title of the histogram.
    """
    fig = px.histogram(
        df,
        x=column,
        nbins=nbins,
        title=title,
        labels={column: 'Words in Description', 'count': 'Frequency'}
    )

    fig.update_layout(
        yaxis_title="Count",
        xaxis_title="Number of Words in Description",
        template="plotly_white"
    )

    fig.show()
