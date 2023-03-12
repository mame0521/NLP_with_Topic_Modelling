# NLP_with_Topic_Modelling

# Document Clustering with Topic Modeling

Topic Modeling was applied to cluster unlabeled documents into different classes and identified  latent topics in this project.

## Contents

# Part 1: Data preprocessing 


```python
# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import random
import re

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.decomposition import PCA, KernelPCA
from sklearn.decomposition import LatentDirichletAllocation
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

random.seed(20220522)
```


```python
# Load data into dataframe
df = pd.read_csv("Review_data.csv", sep=',', header=0)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review_body</th>
      <th>star_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Good luck finding a reasonably priced band rep...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>No i get dark on the first week with me!! I wi...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>I dont know if mine was a mistake but it clear...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The rod that holds the bracelet broke several ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>I bought 2 watches , one watch doesnot work at...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

</div>



```python
# Cheching  missing value
df.isnull().sum()
```



    review_body    0
    star_rating    0
    dtype: int64




```python
df.star_rating.value_counts()
```



    5    3000
    4    3000
    3    3000
    2    3000
    1    3000
    Name: star_rating, dtype: int64


```python
# Take only the review_body column 

data = df.loc[:, 'review_body'].tolist()
print(type(data))
print(len(data))
```

    <class 'list'>
    15000



```python
# Take a look at some of the reviews
for _ in range(5):
    print(data[_],"\n")
```

    Good luck finding a reasonably priced band replacement. I ordered the band from the dealer who sold it to me (no one else in town could get one) and Skagen sent the wrong one.  I guess I'll try again, but not allowing anyone else to make bands for your unique watch design seems stupid. I will certainly never buy one again. 
    
    No i get dark on the first week with me!! I will never buy this item and i had buy 5 of them 
    
    I dont know if mine was a mistake but it clearly states aqua so im confused why mine is lime green. I hate lime green and am very irritated. This is why people hate ordering on amazon. Ive spent 100s of dollars on here latey and this one will make me not want to order.  At least its not much money. Just annoying thinking u ordered something and get something else.  Well its going in the trash... 
    
    The rod that holds the bracelet broke several times and the company do not fix it, it is sitting on the drawer so I can come to see the Jeweler to try to fix one more time. Don't buy it. Really. Don't buy it, It is headache. 
    
    I bought 2 watches , one watch doesnot work at all, other watch  runs, its time slows down 5-10 minutes backward. Outwardly the watches look beautiful, it doesnot show time . I don't know why you are selling these kind of watches online. It is a waste my money and time that I bought these watches. 


​    

# Part 2: Tokenize and Stem


```python
# Use nltk's English stopwords.
stopwords = stopwords.words('english')

print("We use " + str(len(stopwords)) + " stop-words from nltk library.")
```

 use 179 stop-words from nltk library.


Use tokenize and stem functions to analyze the reviews.


```python
def tokenization_and_stemming(text):
    '''
    INPUT
    text - string
    OUTPUT
    clean_tokens - a list of words
    This function processes the input using the following steps :
    1. Remove punctuation characters
    2. Tokenize text into list
    3. Stem, Normalize and Strip each word
    4. Remove stop words
    '''
    # Remove punctuation characters and numbers
    text = re.sub(r"[^a-zA-Z]", " ", text)
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Create a instance of stem class
    stemmer = SnowballStemmer("english")
    
    clean_tokens = []
    for word in tokens:
        clean_tok = stemmer.stem(word).lower().strip()
        if clean_tok not in stopwords:
            clean_tokens.append(clean_tok)

    return clean_tokens
```


```python
tokenization_and_stemming(data[42])
```




    ['warranti', 'card', 'box']



# Part 3: Term Frequency-Inverse Document Frequency(TF-IDF)

create the tf-idf matrix using the TfidfVectorizer()


```python
tfidf_model = TfidfVectorizer(
    max_df=0.99, # max_df : maximum document frequency for the given word
    max_features=1000, # max_features: maximum number of words
    min_df=0.01, # min_df : minimum document frequency for the given word
    use_idf=True, # use_idf: if not true, we only calculate tf
    tokenizer=tokenization_and_stemming,
    ngram_range=(1,1) # ngram_range: (min, max), eg. (1, 2) including 1-gram, 2-gram
)


# Fit the TfidfVectorizer to our data
tfidf_matrix = tfidf_model.fit_transform(data) 


print("In total, there are {} reviews and {} terms.".format(
    str(tfidf_matrix.shape[0]), str(tfidf_matrix.shape[1])
))
```

    In total, there are 15000 reviews and 445 terms.



```python
# Check the parameters
tfidf_model.get_params()
```



    {'analyzer': 'word',
     'binary': False,
     'decode_error': 'strict',
     'dtype': numpy.float64,
     'encoding': 'utf-8',
     'input': 'content',
     'lowercase': True,
     'max_df': 0.99,
     'max_features': 1000,
     'min_df': 0.01,
     'ngram_range': (1, 1),
     'norm': 'l2',
     'preprocessor': None,
     'smooth_idf': True,
     'stop_words': None,
     'strip_accents': None,
     'sublinear_tf': False,
     'token_pattern': '(?u)\\b\\w\\w+\\b',
     'tokenizer': <function __main__.tokenization_and_stemming(text)>,
     'use_idf': True,
     'vocabulary': None}



Save the terms identified by TF-IDF.


```python
# Words
tf_selected_words = tfidf_model.get_feature_names()
```


```python
tfidf_matrix
```




    <15000x445 sparse matrix of type '<class 'numpy.float64'>'
    	with 238491 stored elements in Compressed Sparse Row format>



# Part 4: clustering using K-means 

find out possible clusters using  K-means algorithm. 


```python
# Number of clusters
num_clusters = (2,3,5)

for num in num_clusters:
    kmeans = KMeans(n_clusters=num)

    model_km = SilhouetteVisualizer(kmeans, colors='yellowbrick')
    model_km.fit(tfidf_matrix) # Fit the data to the visualizer
           
    model_km.show()
```


![png](E:\NLP and Topic modeling\output_25_0.png)



![png](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/output_25_1.png)



![png](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/output_25_2.png)


the dataset contains the product reviews, which would fall into one of positive, neutral or negative clusters. So select 3 as the number of clusters in the kmeans.


```python
kmeans_model = KMeans(n_clusters=3)

kmeans_model.fit(tfidf_matrix) # Fit the data
```




    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
           n_clusters=3, n_init=10, n_jobs=None, precompute_distances='auto',
           random_state=None, tol=0.0001, verbose=0)



## 4.1. Analyze K-means Result


```python
kmeans_results = df.rename({'review_body':'review'})
clusters = kmeans_model.labels_.tolist()
kmeans_results['cluster'] = clusters
```


```python
kmeans_results.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review_body</th>
      <th>star_rating</th>
      <th>cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Good luck finding a reasonably priced band rep...</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>No i get dark on the first week with me!! I wi...</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>I dont know if mine was a mistake but it clear...</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The rod that holds the bracelet broke several ...</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>I bought 2 watches , one watch doesnot work at...</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>This watch would have been fantastic, if it ha...</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>I have this watch. It looks and feels heavy du...</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>What the hell! I just got the watch today but ...</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>I am mechanically inclined but cannot get this...</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>It didnt work right out from the box. I had to...</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

</div>




```python
print ("Number of reviews included in each cluster:")
cluster_size = kmeans_results['cluster'].value_counts().to_frame()
cluster_size
```

    Number of reviews included in each cluster:





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12612</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>888</td>
    </tr>
  </tbody>
</table>

</div>




```python
kmeans_results.groupby('cluster')['star_rating'].value_counts()
```




    cluster  star_rating
    0        1              2632
             2              2626
             5              2523
             3              2490
             4              2341
    1        4               343
             3               329
             2               296
             1               293
             5               239
    2        4               316
             5               238
             3               181
             2                78
             1                75
    Name: star_rating, dtype: int64



As shown above, that cluster 0 contains more negative reviews, cluster 2 indicates more  positive reviews and the reviews in cluster 1 represent more neutral. 


## 4.2. Plot the clustering result


```python
pca = KernelPCA(n_components=2)
tfidf_matrix_np=tfidf_matrix.toarray()
X = pca.fit_transform(tfidf_matrix_np)

xs, ys = X[:, 0], X[:,1]
```


```python
pca_df = pd.DataFrame(dict(x = xs, y = ys, Cluster = clusters ))
plt.subplots(figsize=(16,9))
sns.scatterplot('x', 'y', data=pca_df, hue='Cluster')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fd1a0ed0990>




![png](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/output_36_1.png)


# Part 5: Topic Modeling - Latent Dirichlet Allocation


```python
# Use LDA for clustering
LDA = LatentDirichletAllocation(n_components=3)
```


```python
# Term frequency for LDA model
tf_lda = CountVectorizer(
    max_df=0.99,
    max_features=500,
    min_df=0.01,
    tokenizer=tokenization_and_stemming,
    ngram_range=(1,1))


tf_matrix_lda = tf_lda.fit_transform(data) 


print ("In total, there are {} reviews and {} terms.".format(
    str(tf_matrix_lda.shape[0]), str(tf_matrix_lda.shape[1])
))
```

    In total, there are 15000 reviews and 445 terms.



```python
print(tf_matrix_lda.shape)
```

    (15000, 445)



```python
# Feature names
lda_feature_name = tf_lda.get_feature_names()
```


```python
# Document topic matrix for tf_matrix_lda
lda_output = LDA.fit_transform(tf_matrix_lda)
print(lda_output.shape)
```

    (15000, 3)



```python
# Topics and words matrix
# Components_[i, j] can be viewed as pseudocount that represents the number of times word j was assigned to topic i.
topic_word = LDA.components_
print(topic_word.shape)
```

    (3, 445)



```python
# Column names
topic_names = ["Topic" + str(i) for i in range(LDA.n_components)]

# Index names
doc_names = ["Doc" + str(i) for i in range(len(data))]

df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topic_names, index=doc_names)

# Get dominant topic for each document
topic = np.argmax(df_document_topic.values, axis=1)
df_document_topic['topic'] = topic

df_document_topic.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Topic0</th>
      <th>Topic1</th>
      <th>Topic2</th>
      <th>topic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Doc0</th>
      <td>0.75</td>
      <td>0.23</td>
      <td>0.01</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Doc1</th>
      <td>0.79</td>
      <td>0.04</td>
      <td>0.16</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Doc2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>0.01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Doc3</th>
      <td>0.95</td>
      <td>0.02</td>
      <td>0.03</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Doc4</th>
      <td>0.97</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Doc5</th>
      <td>0.33</td>
      <td>0.48</td>
      <td>0.19</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Doc6</th>
      <td>0.52</td>
      <td>0.13</td>
      <td>0.35</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Doc7</th>
      <td>0.48</td>
      <td>0.49</td>
      <td>0.03</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Doc8</th>
      <td>0.02</td>
      <td>0.95</td>
      <td>0.03</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Doc9</th>
      <td>0.64</td>
      <td>0.33</td>
      <td>0.03</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

</div>




```python
df_document_topic['topic'].value_counts().to_frame()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>topic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>5533</td>
    </tr>
    <tr>
      <th>0</th>
      <td>5478</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3989</td>
    </tr>
  </tbody>
</table>

</div>



The cluster size is more even in this case.


```python
# Topic-word matrix
df_topic_words = pd.DataFrame(LDA.components_)

# Column and index
df_topic_words.columns = tf_lda.get_feature_names()
df_topic_words.index = topic_names

df_topic_words.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>abl</th>
      <th>absolut</th>
      <th>accur</th>
      <th>actual</th>
      <th>adjust</th>
      <th>advertis</th>
      <th>ago</th>
      <th>alarm</th>
      <th>almost</th>
      <th>alreadi</th>
      <th>...</th>
      <th>women</th>
      <th>wore</th>
      <th>work</th>
      <th>worn</th>
      <th>worth</th>
      <th>would</th>
      <th>wrist</th>
      <th>wrong</th>
      <th>year</th>
      <th>yet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Topic0</th>
      <td>162.543364</td>
      <td>160.717966</td>
      <td>28.893088</td>
      <td>87.010639</td>
      <td>36.932329</td>
      <td>15.189714</td>
      <td>259.233996</td>
      <td>0.370564</td>
      <td>136.788735</td>
      <td>243.412807</td>
      <td>...</td>
      <td>47.373512</td>
      <td>274.403581</td>
      <td>2172.071511</td>
      <td>163.298610</td>
      <td>251.509245</td>
      <td>1130.238884</td>
      <td>215.988093</td>
      <td>136.972321</td>
      <td>1240.301625</td>
      <td>118.215664</td>
    </tr>
    <tr>
      <th>Topic1</th>
      <td>25.329756</td>
      <td>44.923534</td>
      <td>221.400551</td>
      <td>239.407167</td>
      <td>129.827761</td>
      <td>163.427593</td>
      <td>35.490527</td>
      <td>0.939582</td>
      <td>88.926291</td>
      <td>53.353420</td>
      <td>...</td>
      <td>15.554201</td>
      <td>5.700975</td>
      <td>512.005305</td>
      <td>89.414305</td>
      <td>285.564113</td>
      <td>966.313557</td>
      <td>58.158623</td>
      <td>121.692383</td>
      <td>228.675000</td>
      <td>136.569037</td>
    </tr>
    <tr>
      <th>Topic2</th>
      <td>155.126880</td>
      <td>4.358500</td>
      <td>156.706361</td>
      <td>249.582195</td>
      <td>489.239911</td>
      <td>52.382693</td>
      <td>2.275477</td>
      <td>433.689854</td>
      <td>201.284974</td>
      <td>63.233773</td>
      <td>...</td>
      <td>120.072287</td>
      <td>5.895444</td>
      <td>507.923184</td>
      <td>87.287085</td>
      <td>47.926642</td>
      <td>881.447559</td>
      <td>1777.853284</td>
      <td>6.335296</td>
      <td>139.023376</td>
      <td>69.215299</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 445 columns</p>

</div>




```python
# Print top n keywords for each topic
def print_topic_words(tfidf_model, lda_model, n_words):
    words = np.array(tfidf_model.get_feature_names())
    topic_words = []
    # For each topic, we have words weight
    for topic_words_weights in lda_model.components_:
        top_words = topic_words_weights.argsort()[::-1][:n_words]
        topic_words.append(words.take(top_words))
    return topic_words

topic_keywords = print_topic_words(tfidf_model=tf_lda, lda_model=LDA, n_words=20)        

df_topic_words = pd.DataFrame(topic_keywords)
df_topic_words.columns = ['Word '+str(i) for i in range(df_topic_words.shape[1])]
df_topic_words.index = ['Topic '+str(i) for i in range(df_topic_words.shape[0])]
df_topic_words
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Word 0</th>
      <th>Word 1</th>
      <th>Word 2</th>
      <th>Word 3</th>
      <th>Word 4</th>
      <th>Word 5</th>
      <th>Word 6</th>
      <th>Word 7</th>
      <th>Word 8</th>
      <th>Word 9</th>
      <th>Word 10</th>
      <th>Word 11</th>
      <th>Word 12</th>
      <th>Word 13</th>
      <th>Word 14</th>
      <th>Word 15</th>
      <th>Word 16</th>
      <th>Word 17</th>
      <th>Word 18</th>
      <th>Word 19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Topic 0</th>
      <td>watch</td>
      <td>love</td>
      <td>one</td>
      <td>work</td>
      <td>time</td>
      <td>batteri</td>
      <td>band</td>
      <td>bought</td>
      <td>year</td>
      <td>replac</td>
      <td>day</td>
      <td>would</td>
      <td>get</td>
      <td>look</td>
      <td>wear</td>
      <td>month</td>
      <td>got</td>
      <td>back</td>
      <td>great</td>
      <td>return</td>
    </tr>
    <tr>
      <th>Topic 1</th>
      <td>br</td>
      <td>watch</td>
      <td>look</td>
      <td>good</td>
      <td>price</td>
      <td>time</td>
      <td>qualiti</td>
      <td>one</td>
      <td>veri</td>
      <td>like</td>
      <td>product</td>
      <td>would</td>
      <td>great</td>
      <td>get</td>
      <td>case</td>
      <td>cheap</td>
      <td>day</td>
      <td>buy</td>
      <td>amazon</td>
      <td>made</td>
    </tr>
    <tr>
      <th>Topic 2</th>
      <td>watch</td>
      <td>veri</td>
      <td>look</td>
      <td>band</td>
      <td>like</td>
      <td>nice</td>
      <td>wrist</td>
      <td>time</td>
      <td>face</td>
      <td>light</td>
      <td>small</td>
      <td>use</td>
      <td>read</td>
      <td>big</td>
      <td>easi</td>
      <td>littl</td>
      <td>good</td>
      <td>color</td>
      <td>size</td>
      <td>would</td>
    </tr>
  </tbody>
</table>

</div>




```python
df_document_topic["star_rating"] = df.star_rating.values
```


```python
df_document_topic.groupby('topic')['star_rating'].value_counts()
```




    topic  star_rating
    0      1              1715
           2              1149
           5              1092
           3               822
           4               700
    1      5               907
           4               808
           3               774
           1               758
           2               742
    2      4              1492
           3              1404
           2              1109
           5              1001
           1               527
    Name: star_rating, dtype: int64



the result above show that topic 0 is more like a negative topic, topic 1 and 2 seem to be positive topic.

# Part 6: Conclusion

The experiment results show:
1) K-means is very sensitive to outliers and could produce very small incorrect clusters, and also works badly with clusters of different sizes and densities.
2) LDA is a generative statistical model that allows sets of observations to be explained by unobserved classes. In this task, LDA works better than K-means.


```python

```
