---
layout: default
---

# 1. Abstract
Tribe Dynamics is a San Francisco-based consulting company that measures social media engagement for beauty, fashion and lifestyle brands. The company provides software, data, and consulting services to some of the most well-known brands, private equity firms and Fortune 500 companies in the space. Tribe initiated its business from English data, since the company started from tracking brands in the U.S. As the company expands internationally to Europe and Asia, it must be able to build the same classification tools for other languages, aiming to the same performances achieved in the English language, while also taking advantage of its wide amount of English training data. In this study, we present two approaches that we used to try to solve this problem. The first one, through neural network language modeling, learns monolingual word embedding representations which, through an appropriate transformation, can be mapped to a shared vector space, thus producing multilingual word vectors that can be used for any text classification purpose.
The second one, addressing the task of cross-lingual text classification, trains a classification model in the source language and ports the model to the target language, with the translation knowledge learned using an Expectation Maximization algorithm, resulting in a very effective way to take advantage of Tribe's conspicuous availability of English data.


# 2. Scope and Project Outline


# 3. Methodologies

## 3.1 Bilingual Word Embeddings

### 3.1.1 Data Preprocessing

### 3.1.2 Word Embeddings
#### 3.1.2.1 Neural Network Language Modeling
#### 3.1.2.2 Alignment

### 3.1.3 Performance Evaluation
#### 3.1.3.1 Binomial Text Classification
#### 3.1.3.2 Visualization Tool

## 3.2 Cross-Lingual Latent Model

# 4. Conclusion and Future Work



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
