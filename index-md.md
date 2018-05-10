---
layout: default
---

# Abstract
Tribe Dynamics is a San Francisco-based consulting company that measures social media engagement for beauty, fashion and lifestyle brands. The company provides software, data, and consulting services to some of the most well-known brands, private equity firms and Fortune 500 companies in the space. Tribe initiated its business from English data, since the company started from tracking brands in the U.S. As the company expands internationally to Europe and Asia, it must be able to build the same classification tools for other languages, aiming to the same performances achieved in the English language, while also taking advantage of its wide amount of English training data. In this study, we present two approaches that we used to try to solve this problem. The first one, through neural network language modeling, learns monolingual word embedding representations which, through an appropriate transformation, can be mapped to a shared vector space, thus producing multilingual word vectors that can be used for any text classification purpose.
The second one, addressing the task of cross-lingual text classification, trains a classification model in the source language and ports the model to the target language, with the translation knowledge learned using an Expectation Maximization algorithm, resulting in a very effective way to take advantage of Tribe's conspicuous availability of English data.

# 1. Introduction
Tribe Dynamics is a consulting company that provides measurements of social media engagement for beauty, fashion and lifestyle brands. They have been consulting, mostly in the American market, for the last four years, providing web-based tools and periodic reports to their clients, to track their social media engagement. Their marquee offering is a metric called Earned Media Value (EMV) which is derived out of various engagement markers (e.g. likes, comments, shares, etc.) collected from their clients’ social media posts.

To calculate these metrics, they collect textual posts by scraping popular social media platforms (e.g. Facebook, Twitter, Instagram, Youtube, etc.) using customized search phrases that are provided by their in-house experts. Once these posts have been collected and appropriately cleaned, it is imperative to filter them, identifying if a post is talking about a particular brand or a particular class of product. Tribe has been highly successful in this task with American brands, whose posts are mostly in English, with their proprietary classification model.

Over the last couple of years, they have been rapidly expanding into the European and Asian markets. Naturally, the language repertoire of their posts has been growing accordingly. Their model does not perform with the same efficiency, achieved on the English text classification, across these new languages. In particular, their model needs sufficient training data in these new languages before being able to train correctly. This problem is at the center of what our project tries to accomplish - provide a model that scales well across different languages with limited amount of training data, while exploiting the structural similarity across languages.

# 2. Scope and Project Outline
The fundamental goal of the project is to improve upon Tribe’s proprietary classification model to essentially develop a brand ~~and vertical~~ classification model that scales well across languages with limited amount of training data. The model is expected to learn and adapt the latent linguistic structure, present in the social media domain, across different languages. To this extent, we investigated two approaches that could take on this challenging task. First, we trained multi-language aligned word embedding representations, to evaluate a technique which aims at overcoming the cumbersomeness of the n-grams representation which is currently in place in Tribe's proprietary model. Additionaly, we developed an evaluation framework of these word embeddings representation, through a web-based visualization tool that serves as a facilitator for Tribe's developers to join the human-in-the-loop pipeline to directly assess the quality of the representation and devise where to act in case something needs to be improved. Lastly, by means of a cross-lingual latent generative model, we demonstrate an approach that shows how Tribe could transfer its model's knowledge, developed on its rich English training data, across other languages, while having to maintain only one model, leveraging its generative nature that does not require a lot of training data in the target language.

# 3. Procedures and methods
## 3.1 Data Exploration
### 3.1.1 Labelled Data
For our study, Tribe provided us 106 datasets (one dataset per brand) in which entries are specified as follows:

| brand_id | brand_name | text | lang | model_decision | labels | mturker | link |
|:---------|:-----------|:-----|:-----|:---------------|:-------|:--------|:-----|
| 8009     | Caudalie   | Favorites Summer Crushes - $30 This is another ...     | en | 1        | [1, 0, 1, 1]                                                        | 1 | https://www.twitter.com/Sephora/statuses/46565...      |

These datasets are used to serve the task of binomial classification, which means determining whether a particular social media post is about a specific brand or not. Each of the entries in the datasets represent a social media post that was gathered querying social media platforms, according to keywords provided by Tribe's in-house experts. Most of these features are self explanatory. Following is a brief description of the less immediate ones:

*   *mturker*: being a supervised classification task, labeling is required. A value of 1 in this feature indicates that the labeling was performed by one (or more) individuals from Amazon MTurk. A value of 0 would indicate that the labeling was performed by one (or more) Tribe's employees.
*   *labels*: list of labels that indicates which class each of the labelers assigned to the provided post. Eventually, these labels will be aggregated into a single value according to majority voting (if a tie is present, it will be mapped to 0 to reduce false positives presence).

In order to provide a more accurate understanding of the pipeline which we wanted to deploy, we decided to focus our attention only on **English and Italian languages**. First, because three components of our group are native Italian speakers, which simplifies the process of manual evaluation of the provided text. Furthermore, Tribe wants to expand internationally to Europe. Italian is a Latin language, so it shares with Spanish and French - two of the most common idioms of the European continent - the same linguistic structure, thus enabling the possibility of transfering the same ideas that were applicable in Italian, to the other mentioned languages.

It is important to notice that, on this reduced datasets, the disparity between the number of English training instances and Italian ones is substantial. Moreover, we identified a considerable class imbalance among all datasets, which greately impacts the performance of the classification and will force us to use other performance metrics, instead of accuracy, to evaluate the results.

### 3.1.2 Unlabelled Data
For the purpose of training the word embeddings, Tribe also provided us two large sized datasets (7'739'316 instances for the Italian dataset, 9'154'680 for the English one), which gathered a set of scraped social media posts, without any data preprocessing or labeling previously performed on them. The csv files simply look like this:

| index | post_message                                       |
|:------|:---------------------------------------------------|
| 0     | Un bellissimo scatto del mio amico fashion_pro...  |
| 1     | Uno staff da paura 😍😍 #evento #moda #tourleade... |

[comment]: <> (Italian: 2.0 GB)
[comment]: <> (English: 1.2 GB)

To make the computations feasible and reduce the amount of noise present in the data, we had to perform some data cleaning procedures, which we will detail in this upcoming chapter.

## 3.2 Data Cleaning
Before performing the cleaning of each post in the unlabelled datasets, we perform a sentence tokenization of the entire corpus, through the NTLK sentence tokenizer. This facilitates the cleaning process with regular expressions and avoids merging together multiple sentences, after punctuation is removed, which would result in an undesired behavior if we want to use this data for language modeling purposes.


Inspired by the [preprocessing Perl script](https://github.com/facebookresearch/fastText/blob/master/wikifil.pl) that was used to train the FastText Wikipedia embeddings - which we will detail later on - each of the sentence tokenized posts is processed with the following instructions:

```python
# Lowercase string
doc = lowercase_string(doc)
# Replace html entities (like &amp; &lt; ...)
doc = escape_html_entities_from_string(doc)
# Remove HTML tags
doc = remove_regexp_from_string(doc, r"<[^>]*>", " ")
# Remove URL links
doc = remove_regexp_from_string(doc, r"http\S+", " ")
# Strip off punctuation (except: ' - / _ )
doc = remove_regexp_from_string(doc, r"[!\"#$%&()*+,.:;<=>?@\[\]^`{|}~]", " ")
# Remove multiple occurrences of the only non alphabetic characters that we kept
doc = remove_regexp_from_string(doc, r"-{2,}|'{2,}|_{2,}", " ")
# Remove cases of "alone" special characters (like: " - " or " _   ")
doc = remove_regexp_from_string(doc, r"( {1,}- {1,})|( {1,}_ {1,})|( {1,}' {1,})", " ")
# Remove all words that containt characters which are not the ones in this list: a-zàèéìòóù '-_
doc = remove_regexp_from_string(doc, r"[A-Za-zàèéìòóù]*[^A-Za-zàèéìòóù \'\-\_]\S*", " ")
# Clean "mistakes" like: 'word -word _word -> word
doc = remove_regexp_from_string(doc, r"(^| )[(\')|(\-)|(\_)]", " ")
# Clean "mistakes" like: word' word- word_ -> word
doc = remove_regexp_from_string(doc, r"[(\')|(\-)|(\_)]($| )", " ")
```

`remove_regexp_from_string(input_string, regexp_str, replacement)` using regular expressions Python's package `re`,replaces each match of the regular expression in the input string, with the replacement parameter, eventually producing a cleaned string that can be used as a building block of the word embeddings training corpus.

## 3.3 Computational Resources
The primary programming language that was used in this project is Python. To process the data, we took advantage of Pandas and NLTK. For statistical modeling, we used scikit-learn and PyTorch, as the deep learning framework to learn the embeddings, because of its strong GPU acceleration capabilities. The web-based visualization tool was built using HTML, CSS and Javascript, by means of its data visualization library D3. Amazon Web Services was used to train the embeddings on the cloud, taking advantage of an NVIDIA GPU. Finally, GitHub was used for project collaboration and version control.

# 4. Mathematical Modeling

## 4.1 Bilingual Word Embeddings

### 4.1.1 Data Preparation
In order for the neural network to learn the embeddings, it has to receive inputs and targets according to a specific format. Because of this, additional operations needed to be performed on the training corpus for the NN to be able to process the data. After having tokenized all the sentences in the corpus, to each of them the `<BOS>` and `<EOS>` tokens were added - respectively at the beginning and at the end of the sentence - to indicate were each sequence was respectively starting and ending. Then, to limit both spatial and temporal computational complexity, we reduced the vocabulary to incorporate only words that had a minimum number of occurrences equal to 10, thus mapping all of the infrequent ones to the `<UNK>` tag. Since training the neural network in batches is a physical necessity - unless we could fit the entire training corpus in memory at once - we had to design a way to reduce all sentences to have the same length. A common approach that is used in the literature is what is called padded sequences. This technique, deployed mostly when dealing with time series data, simply consists in setting the size of each sequence to be as long as the longest sequence in the corpus, thus repeating the concatenation of a padding token - which in our case was `<PAD>` - to the end of each sequence, until it matches the length of the longest sequence availabile. Once we applied this preprocessing step, we realized that the dataset was becoming unnecessarily large. A more accurate analysis, revealed that some of these tokenized posts had a sequence length that was in the order of thousands of words per post, which was totally unrealistic for a sentence tokenized piece of text. Having recognized this problem, we decided to prune sentences that contained more than 52 tokens - including `<BOS>` and `<EOS>` - in order to have the dimensionality of the sequence into a manageable size. Finally, since the neural network accepts only numerical input tensors, each word was mapped to an index value, which simply reflected the postition of the word inside the vocabulary.

[comment]: <> (Esempio di come è una sentence? Come http://deeplearning.net/tutorial/rnnslu.html)

### 4.1.2 Word Embeddings
Word embeddings are dense vectors of real numbers, one for each word in the vocabulary. The purpose of word embeddings is to represent each word to efficiently encode its semantic information, in a way that might be relevant to the task at hand. To compute these dense vectors of real numbers, we decided to learn them through a language modeling task. The details of the approach are presented in the upcoming chapter.

#### 4.1.2.1 Neural Network Language Modeling
A statistical language model is a probability distribution over sequences of words. Given such a sequence, say of length m, it assigns a probability <span><img src="http://latex.codecogs.com/gif.latex?P(w_1,...,w_m)" title="P(w_1,...,w_m)" /></span>
to the whole sequence. In our case, we have trained a Recurrent Neural Network word-level language model. That is, we gave to the RNN - precisely an LSTM (Long Short-Term Memory) - a sequence of words, coming from a social media post, and asked it to model the probability distribution of the next word in the sequence given a sequence of previous words, repeating this process for each post in our training corpus. In terms of inputs and expected outputs, for each padded sequence - detailed in chapter 4.1.1 - the neural network takes as input all the words in the sequence, but the last one, while the target is all the words in the sequence, but the first one. This way, for each word in input, the neural network tries to predict the next word in the sentence in output.

[comment]: <> (Show example here)

Our primary task remains learning word embeddings. Because of this, the LSTM, instead of taking in input a one-hot-encoded version of each word, it is actually fed with 300 dimensional word embeddings. In this way, the NN is also able to tune these parameters so that each word's semantic information is encoded in the vectorial representation, which is the main expectation that we have for our model.

The reason why we used 300 dimensional word embeddings is because we have decided to enhance the quality of our representation with state-of-the-art embeddings coming from FastText, a library for efficient learning of word representations and sentence classification, developed at Facebook.

To do such enhancement, we performed a customization of our neural network setting. Precisely, we turned it into a multi-channel neural network. The idea behind multi-channel neural networks is to split the network in two parts - known as static and dynamic channels - and perform backpropagation only on the dynamic channel, thus keeping unchanged the parameters of the static one. To adapt it to our framework, we set up two embedding layers:
* **Static**: initialized with FastText word embeddings. These parameters will be unchanged for the entire learning process. No backpropagation is performed on them;
* **Dynamic**: initialized with FastText word embeddings, but changed dynamically through backpropagation on each training iteration.

![Multi-channel Neural Network]({{ site.url }}/assets/images/multi-channel-NN.png)

With this approach, in the neural network, each word is represented by two embeddings, one coming from the static embedding and one from the dynamic one. The expected outcome of this approach is to leverage the quality of pre-trained embeddings - learned on Wikipedia text - and adapt it to the fashion and cosmetics domain.

[comment]: <> (Check if there are other things to say here)

Once these word embeddings have been learned separately, both in English and Italian, they are ready to be aligned in the same vector space.

#### 4.1.2.2 Alignment<span></span>
Monolingual word vectors embed language in a high-dimensional vector space, such that the similarity of two words is defined by their proximity in this space. They enable
us to train sophisticated classifiers but they require independent models to be trained for each language. Crucially, training text obtained in one language
cannot improve the performance of classifiers trained in another, unless the text is explicitly translated. Because of this, increasing interest is now focused on bilingual vectors, in which words are aligned by their meaning, irrespective of the language of origin. The idea is that, starting from two sets of word vectors in different languages - in our case, the previously computed English and Italian embeddings - we learn a linear matrix <span><img src="http://latex.codecogs.com/gif.latex?W" title="W" /></span>, trained using a dictionary of shared words between the two languages, to then map word vectors from the "source" language into the "target" language.

As stated above, our method requires a training dictionary of paired vectors, which is used to infer the linear mapping <span><img src="http://latex.codecogs.com/gif.latex?W" title="W" /></span>. Typically this dictionary is obtained by translating common source words into the target language using Google Translate, which was constructed using expert human knowledge. However most European languages share a large number of words composed of identical character strings (e.g. words like "Boston", "DNA", "pizza", etc.). It is probable that identical strings across two languages share similar meanings. Following this reasoning, we extracted these strings to form a "pseudo-dictionary", compiled without any expert bilingual knowledge, containing words that appeared both in the English vocabulary and the Italian vocabulary. We call such words anchor words.

Since euclidean distance is a direct measure of similarity in these mono-lingual embeddings, we want to align embeddings of two languages only using isometric maps, i.e. functions that preserve the relative distance between the words. For instance, rotation is one such transformation where when we rotate the entire embedding cloud about any axis, we do not change the inter-word distances. We formally define this rotation using an orthonormal matrix transformation(Remember orthonormal transformation can rotate the matrix and cannot skew it).  If X is a set of source anchor word embeddings and Y is a set of target anchor word embeddings, then we can define this problem as follows where we minimize the norm of difference between the transformed and target embeddings.

W* = argmin<sub>W in W<sub>O</sub></sub> ||WX-Y||<sub>F</sub> = UV<sup>T</sup> where SVD(XY<sup>T</sup>) = U &Sigma; V<sup>T</sup>
where W<sub>O</sub> is a set of all orthonormal matrices

This problem is famously called the Orthogonal Procrustes Problem which has a neat closed form solution that is just a product of two rotation matrices of a singular value decomposition. Remember that singular value decomposition gives a product of three matrices with U,V being rotation matrices and &Sigma; being the skew matrix. Since our solution is a product of two rotation only matrices, it is in essence a rotation matrix only. Thus, we have achieved the closest possible alignment between the two embeddings without any skewing of inter-word distances in either language.
<!---
NOT NEEDED FROM HERE
Once this "pseudo-dictionary" has been computed, we can find the actual transformation matrix by means of the Singular Value Decomposition (SVD), a linear algebra technique used to factorize a matrix into 3 sub-matrices. In our case, SVD is performed on a square matrix <span><img src="http://latex.codecogs.com/gif.latex?M" title="M" /></span> with the same dimensionality as the word embeddings. This matrix is computed from the product of two ordered matrices - <span><img src="http://latex.codecogs.com/gif.latex?X_D" title="X_D" /></span> and <span><img src="http://latex.codecogs.com/gif.latex?Y_D" title="Y_D" /></span> - formed from the "pseudo-dictionary" - such that the <span><img src="http://latex.codecogs.com/gif.latex?i^{th}" title="i^{th}" /></span> row of {<span><img src="http://latex.codecogs.com/gif.latex?X_D" title="X_D" /></span>,<span><img src="http://latex.codecogs.com/gif.latex?Y_D" title="Y_D" /></span>} corresponds to the source and target language word vectors of the <span><img src="http://latex.codecogs.com/gif.latex?i^{th}" title="i^{th}" /></span> pair in the dictionary. Mathematically wise, it looks like this: <span><img src="http://latex.codecogs.com/gif.latex?M&space;=&space;X_D^{T}Y_D&space;=&space;U\Sigma&space;V^{T}" title="M = X_D^{T}Y_D = U\Sigma V^{T}" /></span>
This SVD step is highly efficient, since <span><img src="http://latex.codecogs.com/gif.latex?M" title="M" /></span> is a square matrix with the same dimensionality as the word vectors. <span><img src="http://latex.codecogs.com/gif.latex?U" title="U" /></span> and <span><img src="http://latex.codecogs.com/gif.latex?V" title="V" /></span> are composed of columns of orthonormal vectors, while <span><img src="http://latex.codecogs.com/gif.latex?\Sigma" title="\Sigma" /></span> is a diagonal matrix containing the singular values.
The last step of the algorithm consists in multiplying back together the two submatrices <span><img src="http://latex.codecogs.com/gif.latex?U" title="U" /></span> and <span><img src="http://latex.codecogs.com/gif.latex?V" title="V" /></span>, discarding <span><img src="http://latex.codecogs.com/gif.latex?\Sigma" title="\Sigma" /></span>, to obtain the final desired linear matrix <span><img src="http://latex.codecogs.com/gif.latex?W" title="W" /></span> needed to perform the mapping from source to target space.
NOT NEEDED TILL HERE
-->

The overall procedure is outlined in the following "pythonic" pseudocode:

```python
def align_embeddings(source_emb_matrix, target_emb_matrix, bilingual_vocab):
    Xd, Yd = keep_rows_in_bilingual_vocab(source_emb_matrix, target_emb_matrix, bilingual_vocab)  # Xd, Yd rectangular matrices, same size
    Xd_T = transpose(Xd)
    M = Xd_T x Yd  # M square matrix
    U, _, V_T = SVD(M)
    W = U x V_T  # W square matrix
    source_aligned_matrix = source_emb_matrix x W
    return source_aligned_matrix
```
It is important to notice that in our case, we performed all of the computations moving from English to Italian, which means that we have always applied the linear transformation <span><img src="http://latex.codecogs.com/gif.latex?W" title="W" /></span> to the English embeddings matrix. Theoretically, given that some words in English can translate to either male or female form of the Italian word, should always be the approach to follow.


### 4.1.3 Performance Evaluation
In this chapter, we outline all the parameters that we have set to perform our evaluation.
- Word Embeddings:
  - Training corpus: aggregation of all the data availabile (labeled and unlabeled data)
    - 1% random sample of the availabile training instances
      - English training instances: 103 840 (word emb. to learn = 192 839)
      - Italian training instances: 117 312 (word emb. to learn = 271 571)
  - Embeddings dimension: 300 (imposed by FastText pre-trained embeddings)
  - LSTM hidden dimensions: 128
  - Training epochs: 10
  - Batch size: 32
  - Loss function: Cross Entropy Loss
  - Optimizer: Adam
    - Learning rate: 0.001
    - Betas: (0.9, 0.999)
    - Eps: 1e-07
- Alignment:
  - Bilingual vocabulary: 20 000 instances


We encountered severe limitations due to to the computing power needed to perform an appropriate training, which forced us to reduce the amount of training instances and the size of the batches, in order for us to come up with a feasible training time for each epoch (i.e. about 1 hour each, for both Italian and English embeddings)


#### 4.1.3.1 Binomial Text Classification
The primary goal of learning a dense compact representation of words with context was to overcome the cumbersomeness and non-scalability of the n-grams representation. To evaluate if this approach is capable of doing so, we had to put them to test into what is one of Tribe's necessities: determine if a social media post, is or is not about a particular brand - also known as binomial classification.

In order for us to compare the two representations equally, we used the same classifier that Tribe is currently using, which is Logistic Regression.

Hence, the comparison has been made with the following two framewords:
- Bag-of-Words + Logistic Regression
- Word Embeddings + Logistic Regression

For the Bag-of-Words representation, only unigrams and bigrams were considered, as adding more grams proved to generate too much sparsity that decreased the performance of the model. To produce the representation, sklearn's CountVectorizer was used (ngrams paramter set to (1,2)).  
Instead, for what concerns the embeddings, the input vector, for each of the training instance of the dataset, was computed summing up all the word embeddings of the words inside the post.

The training data, before being vectorized, has all been cleaned with the same cleaning procedure described in chapter 3.2. Stopwords have not been removed.

The Logistic Regression classifier is sklearn's one, with default parameters.

The performance evaluation is on 20% of the data available for each brand - thanks to the 80-20 train test split - and accuracy was not used as a metric for the evaluation, as the majority of the datasets are extremely imbalanced.

Since the problem is binomial, which means we have to consider one brand at-a-time to evaluate the performance, the following results represent an average performance over all the performances of the classification on each single brand dataset.


**Monolingual Word Embeddings**

| head1        | head two          | three |
|:-------------|:------------------|:------|
| ok           | good swedish fish | nice  |
| out of stock | good and plenty   | nice  |
| ok           | good `oreos`      | hmm   |
| ok           | good `zoute` drop | yumm  |

**Bilingual Word Embeddings**

| head1        | head two          | three |
|:-------------|:------------------|:------|
| ok           | good swedish fish | nice  |
| out of stock | good and plenty   | nice  |
| ok           | good `oreos`      | hmm   |
| ok           | good `zoute` drop | yumm  |

As we can see from these two tables, the performance with the embeddings is comparable on all metrics with the bag-of-words representation. Considering that the embeddings were trained on a subset of the data, there is a lot of room for improvement.

#### 4.1.3.2 Visualization Tool
One of the primary objective of this project was to build a human-in-the-loop pipeline for training a foreign language text classifier. According to Wikipedia, human-in-the-loop is defined as a model that requires human interaction. To accomplish this task, we decided to set up a visualization tool to visualize word embeddings, providing to the user the capability to actively evaluate them.

##### 4.1.3.2.1 Scatter Plot
The major problem of word embeddings, with respect to bag-of-words or other representations, is that it loses interpretability. In fact, each word of the vocabulary is mapped to a high-dimensional vector of apparently random numbers. The purpose of the scatter plot, which we designed, is to try to overcome this interpretability issue, reducing the dimensionality of the 300-dimensional embeddings - which can not be depicted physically - to a 2-dimensional one, by means of a very popular dimensionality reduction technique known as TSNE. Since the vector is now 2-dimensional, it can be put on a xy-plane and similarity between words can be conceived quite easily (e.g. euclidean distance between dot points).

Unfortunately, as it happens with any dimensionality reduction technique, a lot of information is lost along the way. In fact, a consequence of this approach, is that now, what we see in 2 dimensions might not be true in 300 dimensions. Precisely, two words that are close in 2 dimensions might not be that close in 300 dimensions. To warn the user/developer of this, we introduced a color gradient. This color gradient is computed as the absolute delta between the cosine similarity of two words in 2 dimensions and the cosine similarity of two words in 300 dimensions. If a dot color is dark, it means that the delta is small and what the user is seeing is considered as a reliable representation. In the opposite case, the user should be warned that what he is seeing, is true only thanks to the dimensionality reduction.

Lastly, the size of each dot is not given by chance. Logistic regression - which is the classifier that Tribe uses for its classification tasks - if trained with a bag of words representation, is able to output a set of coefficients that represent the influency that each feature (i.e. word) has on the discrimination of an instance to belong to the positive class or not. Along these lines, we decided to set the size of each dot in the scatter plot proportionally to this value. This means that if a word has a lot of influency in the classification, its dot size should be high. Oppositely, if it does not. The purpose of this is to focus the attention of the user to all the words that are most impactful in the classification, thus highlighting in which neighborhood of the plot, the embeddings should behave correctly.

##### 4.1.3.2.2 Line Plot
The second important feature of this visualization tool is the line plot. Once a word is selected, a set of rectangles get plotted on it. Each of this rectangles represents a word in the vocabulary, ordered on this line according to the actual cosine similarity in 300 dimensions that this word has with respect to the selected one. The purpose of this plot is to actually show on an ordered line, which are the most similar words with respect to a selected one, according to its true similarity, since it is computed in 300 dimensions. The color of the rectangles, in this case, does not encode a special meaning. Its purpose is just to relate directly with the dot that represents the same word on the scatter plot, thus not confusing the user with two different colors when considering the same word.

##### 4.1.3.2.3 Screenshots
Following, is a set of screenshots gathered from the actual tool to provide to the reader a quick guide to its usage.

[picture]

[picture]


## 4.2 Model 2 : Mixture of Word Translations
Our earlier claim about Tribe Dynamics' baseline models is that the model did not leverage the knowledge learned in one language on to another. Aligned word embeddings that we saw earlier was able to achieve it through pre-training and aligning embeddings so that all what the classifier learns in English, similar knowledge is encoded into Italian too. 

This model attempts to transfer knowledge by basing itself on this fundamental question.
<b> Why don't we translate all the foreign text to English and use our base classifier's core competency in English ?</b>

Translation at sentence level requires a neural or statistical machine translation system which has operational costs. Given that our baseline classifier uses bag of words assumption which anyway gets rid of the context within sentence, we resorted to word-level translations between Italian(our proof of concept language) and English. But, word level translations sometimes suffer from ambiguity. For instance, the word "Frank" in English could either mean a name or the adjective. Similarly, in our dataset for brand Dove, the word "Dove" could either mean "where" or the brand "Dove" in Italian. So, we need to learn a model first to learn the probability scores of these word level translations between plausible alternatives. 

This model needs a good bilingual lexicon with multiple possible translations of each of our vocabularies and the quality of lexicon and word tokenization is key to the performance of this model. There are two possible alternatives for this lexicon - direct downloadable files if they exist or Python's vocabulary package which has a translation module which is a wrapper around a machine translation query and returns multiple translations of a given word. Thus, we construct lexicons translating both ways as two python dictionary files.

Now consider two phrases in Italian "Amo Dove"(Love Dove - the brand), "Dove Sei"(Where are you?). If the model has no additional information, in both cases it is logical to translate "dove" into dove and where with 50% chance. But, if we know that the phrase is talking about the brand or not taling about the brand, then the translation probabilities vary drastically. So, here the class variable C(brand or not) has useful information that will help the model to translate better. We want to capture this idea using a simple directed graphical model. In this approach, we use a generative model that generates a new document D in our target language(Italian). The document D could be simply viewed as a bag of words based on our naive assumptions. Similarly, let us call the corresponding source document D'. The graphical model could be seen in Figure 

USE THE GRAPHICAL MODEL FIGURE FROM THE POSTER OVERLEAF DOCUMENT.

We can write the following conditional probabilities for inference directly from the graphical models.

<img src="http://latex.codecogs.com/gif.latex?\begin{aligned}&space;P(d)&space;&=&space;\sum_{c}&space;P(c)&space;\sum_{d'}&space;P(d|d',c)&space;P(d'|c)&space;\\[10pt]&space;&=&space;\Large{\sum_c}&space;P(c)&space;\sum_{d'}&space;\prod_{i=1}^l&space;P(w_i|w_i',c)&space;P(w_i',c)&space;\end{aligned}" title="\begin{aligned} P(d) &= \sum_{c} P(c) \sum_{d'} P(d|d',c) P(d'|c) \\[10pt] &= \Large{\sum_c} P(c) \sum_{d'} \prod_{i=1}^l P(w_i|w_i',c) P(w_i',c) \end{aligned}" />

In this approach, we then use Expectation Maximization to perform latent variable inference which involves training an iterative algorithm that oscillates between these two steps.

 1. Expectation - P(w',C|w) = P(C,w',w) / P(w) = P(w|w',C) P(w',C) / &Sigma;<sub>w'</sub> &Sigma;<sub>C</sub>  P(w|w',C) P(w',C)
 2. Maximization - P(w|w', C) = f(w) * P(w',C|w) / &Sigma;<sub>w</sub> f(w) * P(w',C|w)

Once this iterative algorithm reaches convergence, we have learned the much-needed conditional word translation probabilities P(w|w',C) i.e. Given a word in source language and the class, what is the probability of this source word generating a word in the target language over its entire vocabulary ?

Our next step is to do model transfer from one language to another. For the sake of simplicity sake, we are going to use a Max Entropy Classifier/Softmax Regression which degenerates to logistic regression or bianry classification tasks as ours is. We can formally define the classifier as

P(c|D) = &Pi;<sub>w</sub> e <sup> &lambda;<sub>w</sub> f(w,C)</sup> /Z(d) where the deonminator is just a proability normalization term.

Under our bag of words assumption, our model transfer can be derived as the following equation(For a slightly detailed derivation, check the paper "Cross Language Text Classification by Model Translation and Semi-Supervised Learning".)
<img src="http://latex.codecogs.com/gif.latex?\hat{C}&space;=&space;\arg\max_{c&space;\in&space;C}&space;\prod_{i=1}^V&space;\sum_{j=1}^{n_i}P(w_t^{ij}&space;|w_s^i,c)&space;e^{\lambda_{w_s^i}&space;f_t(w_t^{ij},c)}" title="\hat{C} = \arg\max_{c \in C} \prod_{i=1}^V \sum_{j=1}^{n_i}P(w_t^{ij} |w_s^i,c) e^{\lambda_{w_s^i} f_t(w_t^{ij},c)}" />

We have a simple expression that is easy to maximize over our 2 classes. The term that is being maximized works under bag of words assumption(leading to the outer product on each word of the vocabulary) and has a clean interpretation - It is the classification score of each translation of a word weighed by its conditional translation probability that we learned via Expectation Maximization. Thus, we learn a simple inference model that helps us perform model translation via mixture of word translations.

***RESULTS TABLE FROM GOOGLE SLIDES



We can see that our Mixture of Word Translations model is able to perform competitively as compared to the baselines with full English data and very little labeled data in Italian(only for supervision) whereas the baselines struggle relatively with less amount of data. We note that our lexicon is still under-constructed and definitely has a lot of scope for improvement. Thus, we have been able to train a model that is data-efficient and performs knowledge transfer to leverage our baseline's model's core capabilities in English.

Now let us discuss the merits and demerits of this model.
1. It requires only a bilingual lexicon for training.
2. It builds and maintains only one single classifier on English vocabulary, thus preventing the problem of vocabulary explosion when many languages are in play.
3. It is interpretable and adaptible to any classifier.
4. Saves data costs significantly as you do not need labeled data in target language apart from cases where you want further semi-supervised training.

The potential concerns(along with possible extensions) with this model include
1. Training the EM algorithm is time consuming - each iteration is O(V<sup>2</sup>) where V is vocabulary size. We can use sampling based inference or approximate inference such as variational inference for quick and efficient inference in this setting.
2. Generating quality lexicon is not easy. This is because we do not have access to good quality clean data but few common languages has lexicons from other sources that are reliable to an extent and the users can further manually fine-tune the lexicon. Having said that, the lexicon is a one-time cost for each new langauge and the model works seamlessly after getting a good lexicon.







**Mixture of Word Translations**

| head1        | head two          | three |
|:-------------|:------------------|:------|
| ok           | good swedish fish | nice  |
| out of stock | good and plenty   | nice  |
| ok           | good `oreos`      | hmm   |
| ok           | good `zoute` drop | yumm  |

# 5. Conclusion and Future Work

We set out with addressing the main issue that Tribe Dynamics' classification model faces - lot of data and a well-trained model in English posts, little data and hence poorer model in other languages.  We wished to leverage the cross-lingual similarities and build models that are able to transfer knowledge learned in one language to the other so as to boost efficiency of learning new langauges. Also, there were other overheads such as growing bag of words with more languages and expanding vocabularies which we wished to address.

The classifier on aligned word embeddings performs this transfer learning by pre-training embeddings and aligning them such that any useful features for the classifier is learned jointly by English and Italian in the embedded space. It also provides a compact dense representation of word vectors and encodes word similarity in its neighborhood which promotes better context perception of the model. Besides it is flexible with any classifier and thus helps address the concerns we wished to address with Tribe Dynamics' baselines.  On the other hand, the mixture of word translation models performs transfer learning by word translations. Learning conditional translation probabilities from data helps it translate with less ambiguity. Successful word translation from Italian to English naturally takes advantage of the features learned by the classifier in English, thus accomplishing knowledge transfer. Again, it is a single model limited to english vocabulary only, interpretable and flexible across any classifier thus satisfying all the constraints we had with the baseline model. 

When we pit these two alternatives against each other, the classifier with aligned word embeddings is a more scalable solution particularly when the business is looking to rapidly expand its operations across several countries as it is learned end-to-end and in an unsupervised setting without any specific domain knowledge imposed on the model, a critical advantage which the other model cannot guarantee. In today's era of deep learning, such embedding based models can be easily trained and maintained without causing significant operational costs for the business. With all these advantages in perspective, we conclude with recommending the usage of the classifier built on dynamically trained aligned word embeddings for cross lingual text classification.

[Notes:]
- Training on more data rather than more epochs (limited by time and GPU memory)
- Different classifier from Logistic regression might be more effective with word embeddings
- No hyper-parameter tuning  


# References [to fix]
* FastText
* Offline Bilingual Word Vectors, Orthogonal Transformations and Inverted Softmax


<hr>
<hr>
<hr>

# LEGEND

###### TABLE OF CONTENTS:
1.  [Abstract](#1-abstract)
2.  [Introduction](#2-introduction)
3.  [Problem Statement and Methods](#3-problem-statement-and-methods)
4.  [Procedures and Methods](#4-procedures-and-methods)
5.  [Mathematical Modeling](#5-mathematical-modeling)
6.  [Analysis](#6-analysis)
7.  [Conclusion](#7-conclusion)
8.  [Acknowledgements](#8-acknowledgements)
9.  [References](#9-references)

[comment]: <> (https://jekyllrb.com/docs/themes/#overriding-theme-defaults)

Text can be **bold**, _italic_, or ~~strikethrough~~.

[Link to another page](./another-page.html).

There should be whitespace between paragraphs.

There should be whitespace between paragraphs. We recommend including a README, or a file with information about your project.

# Header 1

This is a normal paragraph following a header. GitHub is a code hosting platform for version control and collaboration. It lets you and others work together on projects from anywhere.

## Header 2

> This is a blockquote following a header.
>
> When something is important enough, you do it even if the odds are not in your favor.

### Header 3

```js
// Javascript code with syntax highlighting.
var fun = function lang(l) {
  dateformat.i18n = require('./lang/' + l)
  return true;
}
```

```ruby
# Ruby code with syntax highlighting
GitHubPages::Dependencies.gems.each do |gem, version|
  s.add_dependency(gem, "= #{version}")
end
```

#### Header 4

*   This is an unordered list following a header.
*   This is an unordered list following a header.
*   This is an unordered list following a header.

##### Header 5

1.  This is an ordered list following a header.
2.  This is an ordered list following a header.
3.  This is an ordered list following a header.

###### Header 6

| head1        | head two          | three |
|:-------------|:------------------|:------|
| ok           | good swedish fish | nice  |
| out of stock | good and plenty   | nice  |
| ok           | good `oreos`      | hmm   |
| ok           | good `zoute` drop | yumm  |

### There's a horizontal rule below this.

* * *

### Here is an unordered list:

*   Item foo
*   Item bar
*   Item baz
*   Item zip

### And an ordered list:

1.  Item one
1.  Item two
1.  Item three
1.  Item four

### And a nested list:

- level 1 item
  - level 2 item
  - level 2 item
    - level 3 item
    - level 3 item
- level 1 item
  - level 2 item
  - level 2 item
  - level 2 item
- level 1 item
  - level 2 item
  - level 2 item
- level 1 item

### Small image

![Octocat](https://assets-cdn.github.com/images/icons/emoji/octocat.png)

### Large image

![Branching](https://guides.github.com/activities/hello-world/branching.png)


### Definition lists can be used with HTML syntax.

<dl>
<dt>Name</dt>
<dd>Godzilla</dd>
<dt>Born</dt>
<dd>1952</dd>
<dt>Birthplace</dt>
<dd>Japan</dd>
<dt>Color</dt>
<dd>Green</dd>
</dl>

```
Long, single-line code blocks should not wrap. They should horizontally scroll if they are too long. This line should be long enough to demonstrate this.
```

```
The final element.
```
