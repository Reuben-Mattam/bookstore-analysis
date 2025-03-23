#!/usr/bin/env python
# coding: utf-8

# # Pre-processing

# In[9]:


import requests
import time
import pandas as pd
import re

# Read the CSV files and assign them to DataFrames
books_df = pd.read_csv("BX-Books.csv")


# In[7]:


# Read the CSV file and assign it to a DataFrame
books_df = pd.read_csv("BX-Books.csv")

def get_book_title(isbn):
    # Base URL for Open Library API
    base_url = "https://openlibrary.org/api/books"

    # Constructing the complete URL with the ISBN
    url = f"{base_url}?bibkeys=ISBN:{isbn}&format=json&jscmd=data"

    try:
        # Sending a GET request to the API
        response = requests.get(url)

        # Checking if the request was successful
        if response.status_code == 200:
            data = response.json()

            # Extracting the title if it exists in the response
            if f"ISBN:{isbn}" in data:
                title = data[f"ISBN:{isbn}"].get("title", "Title not found for this ISBN")
                return title
            else:
                return "Title not found for this ISBN"
        else:
            return "Failed to retrieve data from the API"

    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

def update_invalid_titles(df):
    isbn_titles = []

    for index, row in df.iterrows():
        title = row['Book-Title']
        if not is_valid_string(title):
            isbn = row['ISBN']
            new_title = get_book_title(isbn)
            isbn_titles.append({'ISBN': isbn, 'Book-Title': new_title})
            
            # Introduce delay between requests (adjust as per API rate limit)
            time.sleep(1)  # 1 second delay between requests

    # Create a DataFrame from collected ISBN and Book-Title pairs
    isbn_df = pd.DataFrame(isbn_titles)

    # Save the DataFrame to a new CSV file called 'tester.csv'
    isbn_df.to_csv('tester.csv', index=False)

# Function to check if a string contains only alphanumeric/punctuation characters
def is_valid_string(input_str):
    pattern = r'^[\w\s\d.,?!-:;\'\(\)\[\]@#$%&=+/\\>-]*$'
    if re.match(pattern, input_str):
        return True
    else:
        return False

