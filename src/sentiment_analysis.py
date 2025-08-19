import plotly.express as px
import pandas as pd

def plot_emotion_distribution(emotions_df: pd.DataFrame, emotion_labels=None, nbins: int = 50):
    """
    Plots the distribution of emotion scores from a DataFrame in long format.

    Parameters:
    - emotions_df (pd.DataFrame): DataFrame containing emotion scores per book.
    - emotion_labels (list, optional): List of emotion columns to include. Defaults to common emotions.
    - nbins (int, optional): Number of bins for the histogram. Defaults to 50.

    Returns:
    - fig: Plotly figure object of the emotion score distribution.
    """
    if emotion_labels is None:
        emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "suprise", "neutral"]

    # Reshape DataFrame from wide to long format
    emotions_long = emotions_df.melt(
        value_vars=emotion_labels,
        var_name="emotion",
        value_name="score"
    )

    # Plot the distribution
    fig = px.histogram(
        emotions_long,
        x="score",
        color="emotion",
        nbins=nbins,
        title="Distribution of Emotion Scores",
        labels={"score": "Emotion Score", "emotion": "Emotion"}
    )

    fig.update_layout(
        barmode="overlay",
        template="plotly_white"
    )

    return fig

