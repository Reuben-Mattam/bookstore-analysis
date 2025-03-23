#!/usr/bin/env python
# coding: utf-8

# In[117]:


import pandas as pd
import re
import numpy as np
import math


import nltk
from textblob import TextBlob
#nltk.download('vader_lexicon')
#!pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.stats import norm

from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
#!pip install opencv-python-headless
import matplotlib.pyplot as plt
import seaborn as sns

import sys
#!{sys.executable} -m pip install wordcloud

from scipy.stats import pearsonr
import random
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde


# In[2]:


# Read the CSV files and assign them to DataFrames
books_df = pd.read_csv("BX-Books.csv")
ratings_df = pd.read_csv("BX-Ratings.csv")
users_df = pd.read_csv("BX-Users.csv")
ISBN_df = pd.read_csv("tester.csv")


# # Data Pre-processing

# In[3]:


def remove_quotes(df):
    for index, row in df.iterrows():

        if row['Book-Title'].startswith('"') and row['Book-Title'].endswith('"'):

            df.at[index, 'Book-Title'] = row['Book-Title'][1:-1]
    return df


def update_invalid_titles(df, ISBN_df):
    isbn_to_title = dict(zip(ISBN_df['ISBN'], ISBN_df['Book-Title']))

    for index, row in df.iterrows():
        title = row['Book-Title']
        if not is_valid_string(title):
            isbn = row['ISBN']
            valid_titles = df[(df['ISBN'] == isbn) & (df['Book-Title'].apply(is_valid_string))]['Book-Title']
            if not valid_titles.empty:
                new_title = valid_titles.iloc[0]  # Choose the first valid title
                df.at[index, 'Book-Title'] = new_title
            else:
                if isbn in isbn_to_title:
                    new_title = isbn_to_title[isbn]
                    df.at[index, 'Book-Title'] = new_title
    return df


# Function to check if a string contains only alphanumeric/punctuation characters
def is_valid_string(input_str):
    pattern = r'^[\w\s\d.,?!-:;\'\(\)\[\]@#$%&=+/\\>-]*$'
    if re.match(pattern, input_str):
        return True
    else:
        return False
    
# Function to check if a string represents an integer
def is_integer(text):
    try:
        int(text)
        return True
    except ValueError:
        return False

    
# Function to filter out entries with improper book titles, authors, or non-integer values in User-ID, ISBN, or Book-Rating
def filter_improper_entries(df):
    return df[df.apply(lambda x: is_valid_string(x['Book-Title']) and is_valid_string(x['Book-Author']) and 
                                  is_integer(x['User-ID']) and is_integer(x['Book-Rating']), axis=1)]


# In[4]:


# first steps of preprocessing

# Remove the " occasionally found at the end of a country name
users_df["User-Country"] = users_df["User-Country"].str.rstrip('"')
users_df["User-Age"] = users_df["User-Age"].astype("str").str.rstrip('"')

# Convert into correct datatype
users_df["User-Age"] = users_df["User-Age"].astype(float).astype('Int64')

# Remove users with unreasonable ages
users_df = users_df[users_df['User-Age'].between(5, 99)]

books_df = remove_quotes(books_df)

# Update books with improper titles to their correct ones
books_df = update_invalid_titles(books_df, ISBN_df)


# In[5]:


# Function to check if a string represents an integer
def is_integer(text):
    try:
        int(text)
        return True
    except ValueError:
        return False

    
# Function to filter out entries with improper book titles, authors, or non-integer values in User-ID, ISBN, or Book-Rating
def filter_improper_entries(df):
    return df[df.apply(lambda x: is_valid_string(x['Book-Author']) and 
                                  is_integer(x['User-ID']) and is_integer(x['Book-Rating']), axis=1)]

def preprocess_dataframe(df):
    # Remove books with less than 1 rating
    df = df[df.groupby('ISBN')['Book-Rating'].transform('count') >= 1]
    
    # Remove book authors with less than 3 books
    author_counts = df['Book-Author'].value_counts()
    df = df[df['Book-Author'].isin(author_counts.index[author_counts >= 3])]
    
    # Remove book publishers with less than 5 books
    publisher_counts = df['Book-Publisher'].value_counts()
    df = df[df['Book-Publisher'].isin(publisher_counts.index[publisher_counts >= 5])]
    
    # Remove books with year of publication before 1750 and after 2024
    df = df[(df['Year-Of-Publication'] >= 1750) & (df['Year-Of-Publication'] <= 2024)]
    
    # Remove ratings made by users with only 1 rating
    df = df[df.groupby('User-ID')['Book-Rating'].transform('count') > 1]
    
    # Remove ratings where the user gives the same rating for every book
    df = df[df.groupby('User-ID')['Book-Rating'].transform('nunique') > 1]
    
    return df


# In[6]:


# Merge the DataFrames
merged_data = pd.merge(ratings_df, books_df, on='ISBN', how='inner')
merged_data = pd.merge(merged_data, users_df, on='User-ID', how='inner')

print(f'merged length = {len(merged_data)}\n')

# Filter out entries with improper book titles, authors, or non-integer values in User-ID, ISBN, or Book-Rating
cleaned_data = filter_improper_entries(merged_data)
print(f'cleaned length = {len(cleaned_data)}\n')

filtered_df = preprocess_dataframe(cleaned_data)

print(f'filtered_df length = {len(filtered_df)}\n')


# # Analysis

# In[7]:


# Initialize sentiment analyzer
sid = SentimentIntensityAnalyzer()

def calculate_sentiment_scores(title):
    scores = sid.polarity_scores(title)
    return scores

def calculate_textblob_sentiment(title):
    blob = TextBlob(title)
    return blob.sentiment.polarity


# In[8]:


sentiment_scores = filtered_df['Book-Title'].apply(calculate_sentiment_scores)
filtered_df.loc[:, 'Sentiment-Negative'] = sentiment_scores.apply(lambda x: x['neg'])
filtered_df.loc[:, 'Sentiment-Neutral'] = sentiment_scores.apply(lambda x: x['neu'])
filtered_df.loc[:, 'Sentiment-Positive'] = sentiment_scores.apply(lambda x: x['pos'])
filtered_df.loc[:, 'TextBlob-Sentiment'] = filtered_df['Book-Title'].apply(calculate_textblob_sentiment)


print(filtered_df.head(10))


# In[9]:


filtered_df_clean = filtered_df.dropna(subset=['User-Age'])

# Convert 'User-Age' column to numeric type
filtered_df_clean['User-Age'] = pd.to_numeric(filtered_df_clean['User-Age'], errors='coerce')

# Drop rows with non-numeric values in 'User-Age' column
filtered_df_clean = filtered_df_clean.dropna(subset=['User-Age'])

# Histogram for Age Distribution
plt.figure(figsize=(10, 6))
plt.hist(filtered_df_clean['User-Age'], bins=30, color='skyblue', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# Graphing

# In[10]:


publication_year = filtered_df[(filtered_df['Year-Of-Publication'] >= 1700) & (filtered_df['Year-Of-Publication'] <= 2024)]

plt.figure(figsize=(8, 6))
plt.scatter(publication_year['User-Age'], publication_year['Year-Of-Publication'], color='red', alpha=0.5)
plt.title('User-Age vs Year-Of-Publication')
plt.xlabel('User-Age')
plt.ylabel('Year-Of-Publication')
plt.grid(True)
plt.show()


# In[11]:


# Calculate mean book rating and mean sentiment score for each book publisher
grouped_df = filtered_df.groupby('Book-Publisher').agg({
    'Book-Rating': 'mean',
    'TextBlob-Sentiment': 'mean'
}).reset_index()

grouped_df.columns = ['Book_Publisher', 'Mean-Book-Rating', 'Mean-TextBlob-Sentiment']

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(grouped_df['Mean-Book-Rating'], grouped_df['Mean-TextBlob-Sentiment'], color='blue', alpha=0.5)
plt.title('Mean Rating vs. Mean Sentiment Score by Book Publisher')
plt.xlabel('Mean Book Rating')
plt.ylabel('Mean TextBlob Sentiment Score')

# Add labels for each point
#for i, txt in enumerate(grouped_df['Book_Publisher']):
#    plt.annotate(txt, (grouped_df['Mean-Book-Rating'][i], grouped_df['Mean-TextBlob-Sentiment'][i]))

plt.grid(True)
plt.show()


# In[12]:


# Calculate mean book rating and mean sentiment score for each book publisher
grouped_df = filtered_df.groupby('Book-Publisher').agg({
    'Book-Rating': 'mean',
    'Sentiment-Positive': 'mean',
    'Sentiment-Neutral': 'mean',
    'Sentiment-Negative': 'mean'
}).reset_index()

grouped_df.columns = ['Book_Publisher', 'Mean_Book_Rating', 'Mean_Sentiment_Positive', 'Mean_Sentiment_Neutral', 'Mean_Sentiment_Negative']

plt.figure(figsize=(18, 6))

# Plot for mean sentiment positive
plt.subplot(1, 3, 1)
plt.scatter(grouped_df['Mean_Book_Rating'], grouped_df['Mean_Sentiment_Positive'], color='blue', alpha=0.5)
plt.title('Mean Rating vs. Mean Sentiment Positive')
plt.xlabel('Mean Book Rating')
plt.ylabel('Mean Sentiment Positive')
plt.grid(True)

# Plot for mean sentiment neutral
plt.subplot(1, 3, 2)
plt.scatter(grouped_df['Mean_Book_Rating'], grouped_df['Mean_Sentiment_Neutral'], color='green', alpha=0.5)
plt.title('Mean Rating vs. Mean Sentiment Neutral')
plt.xlabel('Mean Book Rating')
plt.ylabel('Mean Sentiment Neutral')
plt.grid(True)

# Plot for mean sentiment negative
plt.subplot(1, 3, 3)
plt.scatter(grouped_df['Mean_Book_Rating'], grouped_df['Mean_Sentiment_Negative'], color='red', alpha=0.5)
plt.title('Mean Rating vs. Mean Sentiment Negative')
plt.xlabel('Mean Book Rating')
plt.ylabel('Mean Sentiment Negative')
plt.grid(True)

plt.tight_layout()
plt.show()


# # Sentiment Analysis
# 

# Age vs Sentiment Score () - define it by children vs young adult vs adult vs elderly
# Location vs Publisher - compare if specific locaitons, i.e languages, use specific publishers
# Age vs Year of Publicaiton (vs potentially sentiment score)
# Age vs Publisher
# 
# potentially, book publishers that are read by all age ranges indiscrimnantly are more likely to be non fiction and vice versa. Using the sentiment score and that idea, we can check to see if its more likely to be non-fiction/fiction. 

# In[13]:


# sentiment rating vs Age Group

def categorize_age(age):
    if 3 <= age <= 16:
        return 'Children'
    elif 17 <= age <= 30:
        return 'Young Adults'
    elif 31 <= age <= 45:
        return 'Middle-aged'
    else:
        return 'Old Adults'

filtered_df['Age-Range'] = filtered_df['User-Age'].apply(categorize_age)

age_range_order = ['Children', 'Young Adults', 'Middle-aged', 'Old Adults']
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

sentiment_columns = ['Sentiment-Positive', 'Sentiment-Neutral', 'Sentiment-Negative']
for i, sentiment_col in enumerate(sentiment_columns):
    ax = axes[i]
    ax.set_title(sentiment_col)
    for age_range in age_range_order:
        group_df = filtered_df[filtered_df['Age-Range'] == age_range]
        ax.bar(age_range, group_df[sentiment_col].mean())

plt.tight_layout()
plt.show()


# # Publisher Analysis

# Book-Publisher Grouping

# In[14]:


def xicor(X, Y, ties=True):
    random.seed(42)
    n = len(X)
    order = np.array([i[0] for i in sorted(enumerate(X), key=lambda x: x[1])])
    if ties:
        l = np.array([sum(y >= Y[order]) for y in Y[order]])
        r = l.copy()
        for j in range(n):
            if sum([r[j] == r[i] for i in range(n)]) > 1:
                tie_index = np.array([r[j] == r[i] for i in range(n)])
                tie_values = r[tie_index] - np.arange(0, sum(tie_index))
                sampled_values = random.sample(list(tie_values), sum(tie_index))
                r[tie_index] = np.array(sampled_values)
        return 1 - n*np.sum( np.abs(r[1:] - r[:n-1]) ) / (2*np.sum(l*(n - l)))
    else:
        r = np.array([sum(y >= Y[order]) for y in Y[order]])
        return 1 - 3 * np.sum( np.abs(r[1:] - r[:n-1]) ) / (n**2 - 1)

def english_speaking_ratio(countries):
    english_speaking_countries = {'usa', 'canada', 'australia', 'ireland', 'new zealand', 'united kingdom'}
    total_countries = len(countries)
    english_speaking_count = sum(1 for country in countries if isinstance(country, str) and country.strip().lower() in english_speaking_countries)
    return english_speaking_count / total_countries if total_countries != 0 else 0

grouped = filtered_df.groupby('Book-Publisher')

average_user_age = grouped['User-Age'].mean()
average_year_of_publication = grouped['Year-Of-Publication'].mean()
most_popular_country = grouped['User-Country'].agg(lambda x: x.value_counts().index[0])
sentiment_counts = grouped[['Sentiment-Positive', 'Sentiment-Negative', 'Sentiment-Neutral']].sum()
most_frequent_sentiment = sentiment_counts.idxmax(axis=1)
average_rating = grouped['Book-Rating'].mean()  # Adding this line to calculate the average rating
english_speaking_ratio_column = grouped['User-Country'].agg(english_speaking_ratio)
textblob_sentiment = grouped['TextBlob-Sentiment'].mean()

publisher_df = pd.DataFrame({
    'Average User-Age': average_user_age,
    'Average Year-Of-Publication': average_year_of_publication,
    'Most Popular Country': most_popular_country,
    'Sentiment-Positive': sentiment_counts['Sentiment-Positive'],
    'Sentiment-Negative': sentiment_counts['Sentiment-Negative'],
    'Sentiment-Neutral': sentiment_counts['Sentiment-Neutral'],
    'Sentiment': most_frequent_sentiment,
    'Average Rating': average_rating,
    'English-Speaking-Ratio': english_speaking_ratio_column,
    'TextBlob-Sentiment': textblob_sentiment
}).reset_index()

print(publisher_df.head(10))


# In[15]:


plt.figure(figsize=(10, 6))
plt.scatter(publisher_df['Average User-Age'], publisher_df['English-Speaking-Ratio'])
plt.title('English-speaking Ratio vs. Average User Age')
plt.xlabel('Average User Age')
plt.ylabel('English-speaking Ratio')
plt.grid(True)
plt.show()
average_age = publisher_df['Average User-Age']
english_speaking_ratio = publisher_df['English-Speaking-Ratio']
print(np.round(xicor(average_age , english_speaking_ratio, True), 4))


plt.figure(figsize=(10, 6))
plt.scatter(publisher_df['English-Speaking-Ratio'], publisher_df['Average Year-Of-Publication'])
plt.title('English-speaking Ratio vs. Average Year of Publication')
plt.xlabel('English-speaking Ratio')
plt.ylabel('Average Year of Publication')
plt.grid(True)
plt.show()
average_year = publisher_df['Average Year-Of-Publication']
english_speaking_ratio = publisher_df['English-Speaking-Ratio']
print(np.round(xicor(average_year , english_speaking_ratio, True), 4))

plt.figure(figsize=(10, 6))
plt.scatter(publisher_df['English-Speaking-Ratio'], publisher_df['Average Rating'])
plt.title('Average Book Rating vs. Average English-speaking Ratio')
plt.xlabel('English-speaking Ratio')
plt.ylabel('Average Book Rating')
plt.grid(True)
plt.show()
average_rating = publisher_df['Average Rating']
english_speaking_ratio = publisher_df['English-Speaking-Ratio']
print(np.round(xicor(average_rating , english_speaking_ratio, True), 4))


# In[16]:


sentiment_english_ratio_mean = publisher_df.groupby('Sentiment')['English-Speaking-Ratio'].mean()

# Plotting
plt.figure(figsize=(10, 6))

# Iterate over each sentiment category and plot a histogram
for sentiment, english_ratio in sentiment_english_ratio_mean.items():
    plt.bar(sentiment, english_ratio, label=sentiment)

plt.xlabel('Sentiment')
plt.ylabel('Mean English-Speaking Ratio')
plt.title('Mean English-Speaking Ratio by Sentiment')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(publisher_df['English-Speaking-Ratio'], publisher_df['TextBlob-Sentiment'], alpha=0.5)
plt.title('TextBlob Sentiment vs English Speaking Ratio')
plt.xlabel('English Speaking Ratio')
plt.ylabel('TextBlob Sentiment')
plt.grid(True)
plt.show()


# # Book Analysis

# try get a linear correlation on sentiment score vs english speaking level per book
# setiment positive score vs age (if correlation, compare it to book publishers with higherst average sentiment score and age and use that as a subset to reccomend. try with other things).
# sentiment score vs age vs english speaking ratio -> try individual combination to see
# see if you can reduce noise frmo some of the graphs to see if they are more useful like that
# try to form clusters via removing outliers
# linear regression vs test case -> accuracy score
# distribution of ages - > does that match then what our other stores

# In[17]:


def english_speaking_ratio(countries):
    english_speaking_countries = {'usa', 'canada', 'australia', 'ireland', 'new zealand', 'united kingdom'}
    total_countries = len(countries)
    english_speaking_count = sum(1 for country in countries if isinstance(country, str) and country.strip().lower() in english_speaking_countries)
    return english_speaking_count / total_countries if total_countries != 0 else 0

def create_booklist_df(filtered_df):
    # Calculate average book rating, average user age, and English speaking ratio
    booklist_df = filtered_df.groupby('ISBN').agg({
        'Book-Title': 'first',
        'Book-Author': 'first',
        'Year-Of-Publication': 'first',
        'Book-Publisher': 'first',
        'Book-Rating': ['mean', 'count'],
        'User-Age': 'mean',
        'User-Country': lambda x: english_speaking_ratio(x),
        'Sentiment-Negative': 'first',
        'Sentiment-Neutral': 'first',
        'Sentiment-Positive': 'first',
    }).reset_index()

    # Rename columns
    booklist_df.columns = ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Book-Publisher', 
                           'Average-Book-Rating', 'Number-Of-Ratings', 'Average-User-Age', 
                           'English-Speaking-Ratio', 'Sentiment-Negative', 'Sentiment-Neutral', 
                           'Sentiment-Positive']

    return booklist_df


# In[18]:


booklist_df = create_booklist_df(filtered_df)

ratings = booklist_df['Average-Book-Rating']
num_ratings = booklist_df['Number-Of-Ratings']

# Creating scatterplot
plt.figure(figsize=(10, 6))
plt.scatter(ratings, num_ratings, color='blue', alpha=0.5)

# Adding labels and title
plt.title('Number of Ratings vs Average Rating')
plt.xlabel('Average Rating')
plt.ylabel('Number of Ratings')

# Adding grid
plt.grid(True)

# Displaying plot
plt.show()


# In[49]:


# english speaking ratio vs number of rating
plt.figure(figsize=(10, 6))
plt.scatter(booklist_df['English-Speaking-Ratio'], booklist_df['Number-Of-Ratings'], alpha=0.5)
plt.title('Number of Ratings vs English Speaking Ratio')
plt.xlabel('English Speaking Ratio')
plt.ylabel('Number of Ratings')
plt.grid(True)
plt.show()

filtered_books = booklist_df[(booklist_df['English-Speaking-Ratio'] >= 0.8) & (booklist_df['English-Speaking-Ratio'] < 1)]

# Count the number of books in the filtered DataFrame
num_books = len(filtered_books)

print("Number of books with English speaking ratio between [0.8, 1):", num_books)

total_books = len(booklist_df)

print("Total number of books:", total_books)


# In[113]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the exponential function
def exponential_func(x, a, b):
    return a * np.exp(b * x)

# Filter out data points where English speaking ratio equals 1
non_english = booklist_df[booklist_df['English-Speaking-Ratio'] < 1]
x_data = non_english['English-Speaking-Ratio']
y_data = non_english['Number-Of-Ratings']

# Fit the exponential function to the data
popt, pcov = curve_fit(exponential_func, x_data, y_data)

# Calculate the adjusted R-squared score
residuals = y_data - exponential_func(x_data, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y_data - np.mean(y_data))**2)
r_squared = 1 - (ss_res / ss_tot)
adjusted_r_squared = 1 - (1 - r_squared) * (len(y_data) - 1) / (len(y_data) - 2 - 1)

# Plotting the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, alpha=0.5, label='Data')

# Plotting the exponential regression curve
x_range = np.linspace(min(x_data), max(x_data), 100)
plt.plot(x_range, exponential_func(x_range, *popt), color='red', label='Exponential Regression')

plt.title('Number of Ratings vs English Speaking Ratio (Excluding Ratio = 1)')
plt.xlabel('English Speaking Ratio')
plt.ylabel('Number of Ratings')
plt.grid(True)
plt.legend()
plt.show()

# Number of books with English speaking ratio between [0.8, 1)
non_english = non_english[(non_english['English-Speaking-Ratio'] >= 0.8)]
num_books = len(non_english)
print("Number of books with English speaking ratio between [0.8, 1):", num_books)

# Total number of books
total_books = len(booklist_df)
print("Total number of books:", total_books)

# Print the parameters of the exponential regression curve and adjusted R-squared score
print("Exponential regression parameters (a, b):", popt)
print("Adjusted R-squared score:", adjusted_r_squared)


# In[116]:


# Define the parabolic function
def parabolic_func(x, a, b, c):
    return a * x**2 + b * x + c

non_english = booklist_df[booklist_df['English-Speaking-Ratio'] < 1]
non_zero_english = non_english[non_english['English-Speaking-Ratio'] > 0]
x_data = non_zero_english['English-Speaking-Ratio']
y_data = non_zero_english['Number-Of-Ratings']

Q1 = np.percentile(y_data, 25)
Q3 = np.percentile(y_data, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

non_outliers = non_zero_english[(non_zero_english['Number-Of-Ratings'] >= lower_bound) & (non_zero_english['Number-Of-Ratings'] <= upper_bound)]
x_non_outliers = non_outliers['English-Speaking-Ratio']
y_non_outliers = non_outliers['Number-Of-Ratings']



popt, pcov = curve_fit(parabolic_func, x_non_outliers, y_non_outliers)
y_predicted = parabolic_func(x_non_outliers, *popt)
residuals = y_non_outliers - y_predicted


ss_res = np.sum(residuals**2)
ss_tot = np.sum((y_non_outliers - np.mean(y_non_outliers))**2)
r_squared = 1 - (ss_res / ss_tot)


plt.figure(figsize=(10, 6))
plt.scatter(x_non_outliers, y_non_outliers, alpha=0.5, label='Non-Outlier Data')
x_range = np.linspace(min(x_non_outliers), max(x_non_outliers), 100)
plt.plot(x_range, parabolic_func(x_range, *popt), color='red', label='Parabolic Regression')

plt.title('Number of Ratings (Non-Outliers, Non-Zero English Ratio) vs English Speaking Ratio')
plt.xlabel('English Speaking Ratio')
plt.ylabel('Number of Ratings')
plt.grid(True)
plt.legend()
plt.show()

print("Parabolic regression parameters (a, b, c):", popt)
print("R-squared score:", r_squared)


# In[109]:


# Calculate R-squared
residuals = y_data - exponential_func(x_data, *popt)
SSR = np.sum(residuals ** 2)
SST = np.sum((y_data - np.mean(y_data)) ** 2)
R_squared = 1 - (SSR / SST)

# Print the strength and direction of the relationship
if popt[1] > 0:
    print("The relationship between the number of ratings and English-speaking ratio is positive.")
else:
    print("The relationship between the number of ratings and English-speaking ratio is negative.")

# Print the goodness of fit (R-squared)
print("R-squared (Goodness of fit):", R_squared)


# In[81]:


boxplot_data = booklist_df[(booklist_df['English-Speaking-Ratio'] >= 0.8) & (booklist_df['English-Speaking-Ratio'] < 1)]
boxplot_data_ratio_1 = booklist_df[booklist_df['English-Speaking-Ratio'] == 1]

plt.figure(figsize=(10, 6))

# First boxplot
box1 = plt.boxplot(booklist_df['Number-Of-Ratings'], positions=[1], widths=0.6)
plt.ylabel('Number of Ratings')

# Second boxplot
box2 = plt.boxplot(boxplot_data['Number-Of-Ratings'], positions=[2], widths=0.6)

# Third boxplot
box3 = plt.boxplot(boxplot_data_ratio_1['Number-Of-Ratings'], positions=[3], widths=0.6)


outliers1 = box1['fliers'][0].get_data()[1]
average_outliers1 = sum(outliers1) / len(outliers1)
print(f"Percentage of outliers for first boxplot: {percentage_outliers1:.2f}%")
print(f"Average value of outliers for first boxplot: {average_outliers1:.2f}\n")


outliers2 = box2['fliers'][0].get_data()[1]
average_outliers2 = sum(outliers2) / len(outliers2)
print(f"Percentage of outliers for second boxplot: {percentage_outliers2:.2f}%")
print(f"Average value of outliers for second boxplot: {average_outliers2:.2f}")

lower_quartile = box2['whiskers'][0].get_ydata()[1]
upper_quartile = box2['whiskers'][1].get_ydata()[1]
print("Lower Quartile:", lower_quartile)
print("Upper Quartile:", upper_quartile)
print("\n")


outliers3 = box3['fliers'][0].get_data()[1]
average_outliers3 = sum(outliers3) / len(outliers3)
print(f"Percentage of outliers for third boxplot: {percentage_outliers3:.2f}%")
print(f"Average value of outliers for third boxplot: {average_outliers3:.2f}")

plt.xticks([1, 2, 3], ['All Data', 'English-Speaking Ratio [0.8, 1)', 'English-Speaking Ratio 1'])
plt.title('Comparison of Number of Ratings')
plt.show()


# In[104]:


linear_regression_data = target_region[(target_region['Number-Of-Ratings'] >= 5) & (target_region['Number-Of-Ratings'] <= 39)]

# Extracting the filtered data
x = linear_regression_data['English-Speaking-Ratio']
y = linear_regression_data['Number-Of-Ratings']

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.3)
plt.title('Number of Ratings vs English Speaking Ratio for Number of Ratings between 5 and 39')
plt.xlabel('English Speaking Ratio')
plt.ylabel('Number of Ratings')
plt.grid(True)
plt.show()


# its important to note that the english speaking ratio and number of ratings isnt really a 'continuous' data set when defined under these points, but rather they tend to group and clump at specific points.

# In[99]:


coordinates_df = pd.DataFrame({'x': np.exp(x).flatten(), 'y': np.exp(y).flatten()})

# Saving DataFrame to CSV file
coordinates_df.to_csv('coordinates.csv', index=False)

# Printing success message
print("Coordinates saved to coordinates.csv file.")


# In[53]:


# Data
pts_x = np.array(mixed_location_df['Average-User-Age'])
pts_y = np.array(mixed_location_df['English-Speaking-Ratio'])

# Constants
RESOLUTION = 50
LOCALITY = 2.0

# Calculating grid
dx = max(pts_x) - min(pts_x)
dy = max(pts_y) - min(pts_y)
delta = min(dx, dy) / RESOLUTION
nx = int(dx / delta)
ny = int(dy / delta)
radius = (1 / LOCALITY) * min(dx, dy)

grid_x = np.linspace(min(pts_x), max(pts_x), num=nx)
grid_y = np.linspace(min(pts_y), max(pts_y), num=ny)
x, y = np.meshgrid(grid_x, grid_y, indexing='ij')


def gauss(x1, x2, y1, y2):
    """
    Apply a Gaussian kernel estimation (2-sigma) to distance between points.
    """
    return (
        (1.0 / (2.0 * math.pi))
        * math.exp(
            -1 * (3.0 * math.sqrt((x1 - x2)**2 + (y1 - y2)**2) / radius))**2
        / 0.4)


def _kde(x, y):
    """
    Estimate the kernel density at a given position.
    """
    return sum([
        gauss(x, px, y, py)
        for px, py in zip(pts_x, pts_y)
    ])


kde = np.vectorize(_kde)
z = kde(x, y)

xi, yi = np.where(z == np.amax(z))
max_x = grid_x[xi][0]
max_y = grid_y[yi][0]

# Printing the coordinates of the highest point of density
print("Coordinates of the highest point of density:")
print("X-coordinate:", max_x)
print("Y-coordinate:", max_y)

# Plotting density plot
fig, ax = plt.subplots()
pcm = ax.pcolormesh(x, y, z, cmap='inferno', vmin=np.min(z), vmax=np.max(z))
fig.colorbar(pcm, ax=ax, label='Density')
ax.set_title('Density Plot')
ax.set_xlabel('Average User Age')
ax.set_ylabel('English Speaking Ratio')
fig.set_size_inches(6, 6)
fig.savefig('density.png', bbox_inches='tight')

# Plotting scatter plot with marked point
fig, ax = plt.subplots()
ax.scatter(pts_x, pts_y, marker='+', color='blue', label='Data Points')
ax.scatter(max_x, max_y, marker='+', color='red', s=200, label='Max Density')
ax.set_title('Scatter Plot with Max Density')
ax.set_xlabel('Average User Age')
ax.set_ylabel('English Speaking Ratio')
ax.legend()
fig.set_size_inches(6, 6)
fig.savefig('marked.png', bbox_inches='tight')

plt.show()


# In[22]:


# Regions with high user density might indicate areas where certain age groups are more active in rating books.
# Sparse regions might indicate demographics or geographic areas where fewer users are engaged in rating books.
# peaks in density might highlight specific age-location combinations where there's a strong preference or interest in certain types of books.

# find point of highest density and then compare how these graphs model towards these points ?!


# In[23]:


threshold = 8 # change to need

x_ranges = []
y_ranges = []

for i in range(nx):
    for j in range(ny):
        if z[i, j] > threshold:
            x_min = grid_x[i]
            x_max = grid_x[i+1] if i+1 < nx else max(pts_x)
            y_min = grid_y[j]
            y_max = grid_y[j+1] if j+1 < ny else max(pts_y)
            x_ranges.append((x_min, x_max))
            y_ranges.append((y_min, y_max))

# Printing the ranges of high-density regions
if x_ranges:
    print("X ranges:")
    print(f"Lowest X value: {min(x_ranges, key=lambda x: x[0])[0]}")
    print(f"Highest X value: {max(x_ranges, key=lambda x: x[1])[1]}")
    print()

if y_ranges:
    print("Y ranges:")
    print(f"Lowest Y value: {min(y_ranges, key=lambda y: y[0])[0]}")
    print(f"Highest Y value: {max(y_ranges, key=lambda y: y[1])[1]}")
    print()


# In[24]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Assuming you already have booklist_df DataFrame defined

# Filter out rows with NaN values in relevant columns
threed_df = booklist_df.dropna(subset=['Number-Of-Ratings', 'Average-User-Age', 'English-Speaking-Ratio'])

# Extract the columns of interest
number_of_ratings = threed_df['Number-Of-Ratings']
average_user_age = threed_df['Average-User-Age']
english_speaking_ratio = threed_df['English-Speaking-Ratio']

# Enable interactive plotting
plt.ion()

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the data points
ax.scatter(english_speaking_ratio, average_user_age, number_of_ratings)

# Set labels and title
ax.set_xlabel('English Speaking Ratio')
ax.set_ylabel('Average User Age')
ax.set_zlabel('Number of Ratings')
ax.set_title('3D Scatter Plot of Number of Ratings vs Average User Age vs English Speaking Ratio')

plt.show()


# In[25]:


def calculate_sentiment_and_subjectivity(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment.polarity, sentiment.subjectivity

# Apply the function to each 'Book-Title' and assign the scores to new columns
booklist_df['Sentiment-Score'] = booklist_df['Book-Title'].apply(lambda x: calculate_sentiment_and_subjectivity(x)[0])
booklist_df['Subjectivity-Score'] = booklist_df['Book-Title'].apply(lambda x: calculate_sentiment_and_subjectivity(x)[1])

plt.figure(figsize=(8, 6))
plt.scatter(booklist_df['Sentiment-Score'], booklist_df['Subjectivity-Score'], alpha=0.5)
plt.title('Sentiment Score vs Subjectivity Score')
plt.xlabel('Sentiment Score')
plt.ylabel('Subjectivity Score')
plt.grid(True)
plt.show()


# # User Engagment

# In[27]:


booklist_df['User-Engagement-Score'] = (booklist_df['Average-Book-Rating'] * booklist_df['Number-Of-Ratings']) / booklist_df['Number-Of-Ratings'].sum()




# In[36]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Assuming booklist_df is your DataFrame

# Group by author and calculate average sentiment (taking modulus) and subjectivity
author_sentiment_subjectivity = booklist_df.groupby('Book-Author').agg({'Sentiment-Score': lambda x: abs(x).mean(), 'Subjectivity-Score': 'mean'})

# Filter authors with less than 2 books
author_sentiment_subjectivity = author_sentiment_subjectivity[booklist_df.groupby('Book-Author').size() >= 5]

# Extracting features (X) and target variable (y)
X = author_sentiment_subjectivity[['Sentiment-Score']]
y = author_sentiment_subjectivity['Subjectivity-Score']

# Creating a Linear Regression model
model = LinearRegression()

# Fitting the model
model.fit(X, y)

# Predicting
y_pred = model.predict(X)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Linear regression line')
plt.title('Linear Regression: Average Sentiment vs. Average Subjectivity per Author')
plt.xlabel('Average Sentiment (Absolute)')
plt.ylabel('Average Subjectivity')
plt.legend()
plt.grid(True)
plt.show()

# Printing the coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Calculate R-squared
r_squared = r2_score(y, y_pred)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y, y_pred)

# Calculate Root Mean Squared Error (RMSE)
rmse = mean_squared_error(y, y_pred, squared=False)

print("R-squared:", r_squared)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)


# In[41]:


publisher_stats = booklist_df.groupby('Book-Publisher').agg({'Average-Book-Rating': 'median', 'Sentiment-Score': 'mean'})

# Plotting
plt.figure(figsize=(10, 6))

# Scatter plot with Median Book Rating on x-axis and Average Sentiment Score on y-axis
plt.scatter(publisher_stats['Average-Book-Rating'], publisher_stats['Sentiment-Score'], color='blue', alpha=0.5)

# Adding labels and title
plt.xlabel('Median Book Rating')
plt.ylabel('Average Sentiment Score')
plt.title('Comparison of Median Book Rating and Average Sentiment Score by Publisher')

# Adding grid
plt.grid(True)

# Displaying the plot
plt.show()

