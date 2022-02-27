---
date: 2022-02-13
lastmod: 2022-02-26
linktitle: how-does-word2vec-work 
title: Word2vec, Explained 
pined: true
weight: 1
draft: false
tags: ["word2vec","statistics"]
categories: ["ace seq2seq"]
code: https://yizengli.com/404.html
---

A tutorial for Word2vec with statistical details explained. 

<!--more-->

<!-- 
# A quick introduction!

# What are word vectors representations!
 -->

The word2vec uses many contexts of a word to build up its numerical representation (e.g. a 300 dimensional vector). The core idea is to learning the representation of a word against words in the context. Let's start with a toy example to illustrate this and then go over a little bit theory behind it.

**Example.** Suppose we have a [corpus](https://www.merriam-webster.com/dictionary/corpus) of text containing two sentences:
- I want to buy an Apple iPhone.
- I want to eat an Apple now.

Notice the meaning of Apple will be a brand and a fruit respectively under different contexts that is easy for human to recognize but not for any machine. We will first create a vocabulary table with unique words in the corpus like below, where each word, for simplicity, is represented with 2 dimensional vector $$[\theta_{\text{word},1},\theta_{\text{word},2}]\in\mathbb{R}^2.$$

| Word     | Vector |
| ----------- | ----------- |
| I      | $[\theta_{\text{I},1},\theta_{\text{I},2}]$       |
| want      | $[\theta_{\text{want},1},\theta_{\text{want},2}]$       |
| to      | $[\theta_{\text{to},1},\theta_{\text{to},2}]$       |
| buy      | $[\theta_{\text{buy},1},\theta_{\text{buy},2}]$       |
| eat      | $[\theta_{\text{eat},1},\theta_{\text{eat},2}]$       |
| an      | $[\theta_{\text{an},1},\theta_{\text{an},2}]$       |
| Apple      | $[\theta_{\text{Apple},1},\theta_{\text{Apple},2}]$       |
| iPhone      | $[\theta_{\text{iPhone},1},\theta_{\text{iPhone},2}]$       |
| now      | $[\theta_{\text{now},1},\theta_{\text{now},2}]$       |


The next step is to determine how many contextual information you want, that is controlled by the window size $m$. In this example, let $m=1$ for simplicity. Now at every position of word in the corpus, $t=1,2,\ldots,14$, we will focus on the window with at most $2m+1$ words in it since at the start/end of sentence, the window will be **trimmed** in word2vec.

![word2vec-fig1](/img/word2vec/word2vec-fig1.png)

In the figure shown above, The highlighted band is referred to $\textcolor{BurntOrange}{\text{window}}$, in which the red word is called $\textcolor{red}{\text{center}}$ word and blue word is called $\textcolor{Cerulean}{\text{context}}$. Now 




**The word2vec model.**

Given a corpus of text, 

**The probability $\mbox{P}(\mbox{o} \mid \mbox{c})$.** Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod
tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam,
quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo
consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse
cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non
proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

**The loss function $J(\theta)$.** Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod
tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam,
quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo
consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse
cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non
proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

**Visualization.** Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod
tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam,
quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo
consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse
cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non
proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
