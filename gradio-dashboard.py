# ======================================================
# Semantic Book Recommendation System with Gradio
# ======================================================
# This script:
# 1. Loads book metadata + emotion scores
# 2. Embeds descriptions for semantic search
# 3. Retrieves recommendations based on query, category, and tone
# 4. Provides a Gradio dashboard for user interaction
# ======================================================


import pandas as pd
import numpy as np

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()



# Load book dataset with pre-computed emotions
books = pd.read_csv("./data/books_with_emotions.csv")

# Create a large thumbnail column by obtaining the largest resolution book cover
books["large_thumbnail"] = books["thumbnail"] + "&file=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# Reading the tagged descriptions into the text loader
raw_documents = TextLoader("./data/tagged_description.txt").load()
# Instantiate a character text reader
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
# Apply the text_reader to each of the document
documents = text_splitter.split_documents(raw_documents)
# Convert them into documnet embeddings and store them in a vector database 
db_books = Chroma.from_documents(documents, OpenAIEmbeddings())

#Retrieve semanting recommendations and filtering based on categories and sorting based on the emotional tone

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16
) -> pd.DataFrame:
    
    # Obtain results onto 50 recommendations, and access the isbns by splitting them from the descriptions
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)
    #then limit the recommendations into the 15 

    #apply filering based on the category. A dropdown menu appears, with all or the other four categories
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    #sort by emotion, a dropdown manu will appear
    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Suprising":
        book_recs.sort_values(by="suprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs

#function of what to display on the gradio dashboard
def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split() # if it is more than 30 words it adds ...
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str= f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        
        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Suprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g., A story about forgiveness")
        category_droppdown = gr.Dropdown(choices = categories, label = "Select a category:", value ="All")
        tone_dropdown = gr.Dropdown(choices = tones, label ="Select an emotional tone", value="All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns=8, rows = 2)

    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_droppdown, tone_dropdown],
                        outputs = output)
    
# Run the Gradio app
if __name__ == "__main__":
        dashboard.launch()