from langchain_community.document_loaders import TextLoader # uses the raw text and convert it in the appropriat6e formaat for use
from langchain_text_splitters import CharacterTextSplitter #split the whole text into meaningful chunks
from langchain_openai import OpenAIEmbeddings #converting those chunks into document embeddings
from langchain_chroma import Chroma #store embeddings into vector data base

import pandas as pd


def retrieve_semantic_recommendations(
        query: str,
        db,
        books_df: pd.DataFrame,
        top_k: int = 10,
        k_search: int = 50
) -> pd.DataFrame:
    """
    Retrieve the top semantic book recommendations for a query.

    Parameters
    ----------
    query : str
        The search query.
    db : Chroma or similar vector database
        The vector database containing book embeddings.
    books_df : pd.DataFrame
        The DataFrame with book metadata, including 'isbn13'.
    top_k : int, default=10
        Number of books to return.
    k_search : int, default=50
        Number of nearest neighbors to retrieve from the similarity search.

    Returns
    -------
    pd.DataFrame
        DataFrame of the top recommended books.
    """
    # Perform a similarity search on the vector database using the query, retrieving the top k_search results
    recs = db.similarity_search(query, k=k_search) 
    # Extract the ISBN from each recommended document and store them in a list
    books_list = [int(r.page_content.strip('"').split()[0]) for r in recs]
    # Filter the books DataFrame to only include books with the recommended ISBNs and return the top_k results
    return books_df[books_df["isbn13"].isin(books_list)].head(top_k)
