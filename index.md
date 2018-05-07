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
#### 4.1.2.1 Neural Network Language Modeling
Mettere qui che input e output sono semplicemente la sequenza meno 1
#### 4.1.2.2 Alignment

### 4.1.3 Performance Evaluation
#### 4.1.3.1 Binomial Text Classification
#### 4.1.3.2 Visualization Tool

## 4.2 Cross-Lingual Latent Model

# 5. Conclusion and Future Work

# References
*   FastText




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
