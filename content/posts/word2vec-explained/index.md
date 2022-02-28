---
date: 2022-02-13
lastmod: 2022-02-26
linktitle: word2vec-explained 
title: Word2vec, Explained
pined: true
weight: 1
draft: false
tags: ["word2vec","statistics", "theory"]
categories: ["ace seq2seq"]
image: word2vec-fig1.png
---

A tutorial, focused on Skip-Gram, for Word2vec with statistical details explained. 

<!--more-->

<!-- 
# A quick introduction!

# What are word vectors representations!
 -->

The word2vec uses the context of a word to build up its numerical representation. One of word2vec models, called **Skip-Gram**, is to learn the representation of a word against words appearing in the context. Let's start with a toy example to illustrate this with a little bit theory behind it.

**Example.** Suppose we have a [corpus](https://www.merriam-webster.com/dictionary/corpus) of text containing two sentences:
- I want to buy an Apple iPhone.
- I want to eat an Apple now.

We will first create a vocabulary, denoted as $V$, with unique words in the corpus below, where each word, for simplicity, is represented with 2 dimensional vector; $$[\theta_{\text{word},1},\theta_{\text{word},2}]\in\mathbb{R}^2.$$

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

Herein, let $V$ be a vocabulary set and $\boldsymbol{\theta}$ represent all the model parameters,
\begin{align*}
V&=
\begin{pmatrix}
\text{I}\\\\
\text{want}\\\\
\text{to}\\\\
\text{buy}\\\\
\text{eat}\\\\
\text{an}\\\\
\text{Apple}\\\\
\text{iPhone}\\\\
\text{now}\\\\
\end{pmatrix}&
\boldsymbol{\theta}&=
\begin{bmatrix}
\theta_{\text{I},1}\\\\
\theta_{\text{I},2}\\\\
\theta_{\text{want},1}\\\\
\theta_{\text{want},2}\\\\
\vdots\\\\
\theta_{\text{iPhone},1} \\\\
\theta_{\text{iPhone},2} \\\\
\theta_{\text{now},1}\\\\
\theta_{\text{now},2}\\\\
\end{bmatrix}.
\end{align*}
The next step is to determine how much contextual information you want, which is controlled by the window size $m$. In this example, let $m=1$ for simplicity. Now at every position of word in the corpus, $t=1,2,\ldots,T$ with here $T=14$, we will focus on the window with **at most** $2m+1$ words since at the start/end of sentence, the window will be **trimmed** in word2vec.

![word2vec-fig1](./word2vec-fig2.png)

In the figure shown above, The highlighted band is referred to $\textcolor{BurntOrange}{\text{window}}$, in which the red word is called $\textcolor{red}{\text{center}}$ word and blue word is called $\textcolor{Cerulean}{\text{context}}$.

Suppose we now have the probability of $w_{t-1}$ given $w_t$ and the probability of $w_{t+1}$ given $w_t$;
$$\text{Pr}(w_{t-1}\mid w_t)\text{ and }\text{Pr}(w_{t+1}\mid w_t),$$
A natural way in statistics is to maximize the likelihood function
\begin{align*}
L(\boldsymbol{\theta})
&=\prod\limits_{t=1}^TL_t(\boldsymbol\theta)\\\\
&=\prod\limits_{t=1}^{T}\prod\limits_{-m\le j\le m}^{j\neq0} \text{Pr}(w_{t+j}\mid w_t)\\\\
&=\text{Pr}(w_{2}\mid w_1) \\\\
&=\text{Pr}(w_{1}\mid w_2)\text{Pr}(w_{3}\mid w_2)\\\\
&=\cdots\\\\
&=\text{Pr}(w_{5}\mid w_6)\text{Pr}(w_{7}\mid w_6)\\\\
&=\text{Pr}(w_{6}\mid w_7)\\\\
&=\text{Pr}(w_{9}\mid w_8)\\\\
&=\text{Pr}(w_{8}\mid w_9)\text{Pr}(w_{10}\mid w_9)\\\\
&=\cdots\\\\
&=\text{Pr}(w_{12}\mid w_{13})\text{Pr}(w_{14}\mid w_{13})\\\\
&=\text{Pr}(w_{13}\mid w_{14})
\end{align*}
People in machine learning/deep learning like to minimize a function for no particular reason so we equivalently minimize the negative log-likelihood (divided by a constant is a convention to make it not too large, think about billions of words) and name it loss function $J(\boldsymbol{\theta})$ below
\begin{align*}
J(\boldsymbol{\theta})
&=-\frac{1}{T}\sum_{t=1}^TJ_t(\boldsymbol\theta)\\\\
&=-\frac{1}{T}\sum\limits_{t=1}^T\sum\limits_{-m\le j\le m}^{j\neq0}\log \text{Pr}(w_{t+j}\mid w_j).
\end{align*}
If we have the expression of $\text{Pr}(\cdot\mid\cdot)$ then we simply use (stochastic) gradient descent to minimize $J(\boldsymbol{\theta})$ to find the optimal $\boldsymbol{\theta}$. So, back in word2vec, we want to find a vector for each word so that it is **similar** to vectors of words that appear in similar contexts. As for the similarity of two words, we can use [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) (dot product). Hence, we could somehow convert the dot product to probability we assumed above. Considering the output of dot product could be negative, a natural choice is to use $\exp(\cdot)$ to make it positive and then normalize it to be a probability (that is, ranging from 0 to 1). In this way, $\text{Pr}(\textcolor{Cerulean}{\text{context}}\mid \textcolor{red}{\text{center}})$ is expressed as
$$
\frac{\exp(\textcolor{Cerulean}{\text{context}}^\top\textcolor{red}{\text{center}})}{Z},
$$
where $Z$ is just a normalization factor as follows
$$
Z=\sum\limits_{\text{word}\in\text{V}}\exp(\text{word}^\top\textcolor{red}{\text{center}}).
$$
If we take the $\log(\cdot)$ with respect to $\text{Pr}(\textcolor{Cerulean}{\text{context}}\mid \textcolor{red}{\text{center}})$ because we have $\log(\cdot)$ in our loss function $J(\boldsymbol{\theta})$ too, then $\log\text{Pr}(\textcolor{Cerulean}{\text{context}}\mid \textcolor{red}{\text{center}})$ is expressed as
\begin{align*}
\textcolor{Cerulean}{\text{context}}^\top\textcolor{red}{\text{center}}
-\log(Z).
\end{align*}
Notice herein $\textcolor{red}{\text{center}}^\top\textcolor{red}{\text{center}}$ will happen in $Z$ since $\textcolor{red}{\text{center}}\in V$ too. This will make the gradient messy since it has literally square itself. In word2vec, it suggests to use two representations for each word (one for $\textcolor{Cerulean}{\text{context}}$, one for $\textcolor{red}{\text{center}}$, and average them as the final representation vector at the end) as follows

| Word     | Context Vector | Center Vector |
| ----------- | ----------- | ----------- |
| I | $[u_{\textcolor{Cerulean}{\text{I}},1},u_{\textcolor{Cerulean}{\text{I}},2}]$|$[v_{\textcolor{red}{\text{I}},1},v_{\textcolor{red}{\text{I}},2}]$       |
| want | $[u_{\textcolor{Cerulean}{\text{want}},1},u_{\textcolor{Cerulean}{\text{want}},2}]$|$[v_{\textcolor{red}{\text{want}},1},v_{\textcolor{red}{\text{want}},2}]$       |
| to | $[u_{\textcolor{Cerulean}{\text{to}},1},u_{\textcolor{Cerulean}{\text{to}},2}]$|$[v_{\textcolor{red}{\text{to}},1},v_{\textcolor{red}{\text{to}},2}]$       |
| buy | $[u_{\textcolor{Cerulean}{\text{buy}},1},u_{\textcolor{Cerulean}{\text{buy}},2}]$|$[v_{\textcolor{red}{\text{buy}},1},v_{\textcolor{red}{\text{buy}},2}]$       |
| eat | $[u_{\textcolor{Cerulean}{\text{eat}},1},u_{\textcolor{Cerulean}{\text{eat}},2}]$|$[v_{\textcolor{red}{\text{eat}},1},v_{\textcolor{red}{\text{eat}},2}]$       |
| an | $[u_{\textcolor{Cerulean}{\text{an}},1},u_{\textcolor{Cerulean}{\text{an}},2}]$|$[v_{\textcolor{red}{\text{an}},1},v_{\textcolor{red}{\text{an}},2}]$       |
| Apple | $[u_{\textcolor{Cerulean}{\text{Apple}},1},u_{\textcolor{Cerulean}{\text{Apple}},2}]$|$[v_{\textcolor{red}{\text{Apple}},1},v_{\textcolor{red}{\text{Apple}},2}]$       |
| iPhone | $[u_{\textcolor{Cerulean}{\text{iPhone}},1},u_{\textcolor{Cerulean}{\text{iPhone}},2}]$|$[v_{\textcolor{red}{\text{iPhone}},1},v_{\textcolor{red}{\text{iPhone}},2}]$       |
| now | $[u_{\textcolor{Cerulean}{\text{now}},1},u_{\textcolor{Cerulean}{\text{now}},2}]$|$[v_{\textcolor{red}{\text{now}},1},v_{\textcolor{red}{\text{now}},2}]$       |

Thus the model parameters will be updated to
\begin{align*}
\boldsymbol{\theta}&=
\begin{bmatrix}
v_{\textcolor{red}{\text{I}},1}\\\\
v_{\textcolor{red}{\text{I}},2}\\\\
\vdots\\\\
v_{\textcolor{red}{\text{now}},1} \\\\
v_{\textcolor{red}{\text{now}},2} \\\\
u_{\textcolor{Cerulean}{\text{I}},1}\\\\
u_{\textcolor{Cerulean}{\text{I}},2}\\\\
\vdots\\\\
u_{\textcolor{Cerulean}{\text{now}},1} \\\\ 
u_{\textcolor{Cerulean}{\text{now}},2}
\end{bmatrix}.
\end{align*}
Now we could avoid the square issue mentioned above which is beneficial from the computing perspective (but the original one still can work if you just try to optimize loss function no matter how). Anyhow, we have our probability function $\text{Pr}(\textcolor{Cerulean}{\text{context}}\mid \textcolor{red}{\text{center}})$ expressed as
$$
\frac{\exp(u_{\textcolor{Cerulean}{\text{context}}}^\top v_{\textcolor{red}{\text{center}}}) }{\sum\limits_{\text{word}\in\text{V}}\exp(u_{\text{word}}^\top v_{\textcolor{red}{\text{center}} })},
$$ 
and consequently we could calculate our loss function $J(\boldsymbol{\theta})$ given $\boldsymbol{\theta}$. If so, we could just initilize $\boldsymbol{\theta}$ by some random initialization and use stochastic gradient descent algorithm to update $\boldsymbol{\theta}$ one window at a time. 

> **Remark.** In vanilla gradient descent, updating $\boldsymbol\theta$ until we've seen all words will take a super long time (imagine you have billions of words) so in practice we would like to update $\boldsymbol\theta$ for a batch of windows. It is called stochastic or mini-batch gradient gescent.

Let's get to the derivative part to find the optimal $\boldsymbol\theta$.

**At position/timestep $t=1$**, our one-window loss function will be 
\begin{align*}
J(\boldsymbol\theta)
&=-J_1(\boldsymbol\theta)\\\\
&= -\log\text{Pr}(w_2|w_1)\\\\
&= -\log\text{Pr}(\textcolor{Cerulean}{\text{want}}\mid\textcolor{red}{\text{I}})\\\\
&= -u_{\textcolor{Cerulean}{\text{want}}}^\top v_{\textcolor{red}{\text{I}}} + \log \sum\limits_{\text{word}\in\text{V}}\exp(u_{\text{word}}^\top v_{\textcolor{red}{\text{I}}}).
\end{align*}
Taking first derivative with respect to $v_{\textcolor{red}{\text{I}}}$ yields
\begin{align*}
\frac{\partial}{\partial v_{\textcolor{red}{\text{I}}}}J(\boldsymbol\theta)
&=-u_{\textcolor{Cerulean}{\text{want}}}+\frac{\sum\limits_{\text{x}\in\text{V}}\frac{\partial}{\partial v_{\textcolor{red}{\text{I}}}}\exp(u_{\text{x}}^\top v_{\textcolor{red}{\text{I}}})}{\sum\limits_{\text{word}\in\text{V}}\exp(u_{\text{word}}^\top v_{\textcolor{red}{\text{I}}})}\\\\
&=-u_{\textcolor{Cerulean}{\text{want}}}+\frac{\sum\limits_{\text{x}\in\text{V}}u_{\text{x}}\exp(u_{\text{x}}^\top v_{\textcolor{red}{\text{I}}})}{\sum\limits_{\text{word}\in\text{V}}\exp(u_{\text{word}}^\top v_{\textcolor{red}{\text{I}}})}\\\\
&=-u_{\textcolor{Cerulean}{\text{want}}}+\sum\limits_{\text{x}\in\text{V}}u_{\text{x}}\text{Pr}(x\mid\textcolor{red}{\text{I}})\\\\
&=-(\text{observed}-\text{expectation}).
\end{align*}
Interestingly, the derivative of center word is just the difference between observed context and (conditional) expectation of context.

Taking first derivative with respect to $u_{\textcolor{Cerulean}{\text{want}}}$ yields
\begin{align*}
\frac{\partial}{\partial u_{\textcolor{Cerulean}{\text{want}}}}J(\boldsymbol\theta)
&=-v_{\textcolor{red}{\text{I}}} + \frac{\sum\limits_{\text{x}\in\text{V}}\frac{\partial}{\partial u_{\textcolor{Cerulean}{\text{want}}}}\exp(u_{\text{x}}^\top v_{\textcolor{red}{\text{I}}})}{\sum\limits_{\text{word}\in\text{V}}\exp(u_{\text{word}}^\top v_{\textcolor{red}{\text{I}}})}\\\\
&=-v_{\textcolor{red}{\text{I}}} + \frac{v_{\textcolor{red}{\text{I}}}\exp(u_{\textcolor{Cerulean}{\text{want}}}^\top v_{\textcolor{red}{\text{I}}})}{\sum\limits_{\text{word}\in\text{V}}\exp(u_{\text{word}}^\top v_{\textcolor{red}{\text{I}}})}\\\\
&=-v_{\textcolor{red}{\text{I}}} + v_{\textcolor{red}{\text{I}}}\text{Pr}(\textcolor{Cerulean}{\text{want}}\mid\textcolor{red}{\text{I}})\\\\
&=-v_{\textcolor{red}{\text{I}}}[1-\text{Pr}(\textcolor{Cerulean}{\text{want}}\mid\textcolor{red}{\text{I}})].
\end{align*}

So we will update corresponding parameters in the current window,
\begin{align*}
\begin{bmatrix}
v_{\textcolor{red}{\text{I}},1}\\\\
v_{\textcolor{red}{\text{I}},2}\\\\
u_{\textcolor{Cerulean}{\text{want}},1}\\\\
u_{\textcolor{Cerulean}{\text{want}},2}
\end{bmatrix}
&=
\begin{bmatrix}
v_{\textcolor{red}{\text{I}},1}\\\\
v_{\textcolor{red}{\text{I}},2}\\\\
u_{\textcolor{Cerulean}{\text{want}},1}\\\\
u_{\textcolor{Cerulean}{\text{want}},2}
\end{bmatrix}-\alpha
\begin{bmatrix}
\frac{\partial}{\partial v_{\textcolor{red}{\text{I}}}}J(\boldsymbol\theta)\\\\
\frac{\partial}{\partial v_{\textcolor{red}{\text{I}}}}J(\boldsymbol\theta)\\\\
\frac{\partial}{\partial u_{\textcolor{Cerulean}{\text{want}}}}J(\boldsymbol\theta)\\\\
\frac{\partial}{\partial u_{\textcolor{Cerulean}{\text{want}}}}J(\boldsymbol\theta)
\end{bmatrix}
\end{align*}
**At position/timestep $t=2$**, our one-window loss function will be
\begin{align*}
J(\boldsymbol\theta) 
=&-J_2(\boldsymbol\theta)\\\\
=& -\log\text{Pr}(w_1|w_2)-\log\text{Pr}(w_3|w_2)\\\\
=& -\log\text{Pr}(\textcolor{Cerulean}{\text{I}}\mid\textcolor{red}{\text{want}})-\log\text{Pr}(\textcolor{Cerulean}{\text{to}}\mid\textcolor{red}{\text{want}})\\\\
=&-u_{\textcolor{Cerulean}{\text{I}}}^\top v_{\textcolor{red}{\text{want}}} + \log \sum\limits_{\text{word}\in\text{V}}\exp(u_{\text{word}}^\top v_{\textcolor{red}{\text{want}}})\\\\
&-u_{\textcolor{Cerulean}{\text{to}}}^\top v_{\textcolor{red}{\text{want}}} + \log \sum\limits_{\text{word}\in\text{V}}\exp(u_{\text{word}}^\top v_{\textcolor{red}{\text{want}}})\\\\
=&-u_{\textcolor{Cerulean}{\text{I}}}^\top v_{\textcolor{red}{\text{want}}} -u_{\textcolor{Cerulean}{\text{to}}}^\top v_{\textcolor{red}{\text{want}}}\\\\
&+ 2\log \sum\limits_{\text{word}\in\text{V}}\exp(u_{\text{word}}^\top v_{\textcolor{red}{\text{want}}})
\end{align*}
Taking first derivative with respect to $v_{\textcolor{red}{\text{want}}}$ yields
\begin{align*}
\frac{\partial}{\partial v_{\textcolor{red}{\text{want}}}}J(\boldsymbol\theta)
=&-u_{\textcolor{Cerulean}{\text{I}}}+\sum\limits_{\text{x}\in\text{V}}u_{\text{x}}\text{Pr}(x\mid\textcolor{red}{\text{want}})\\\\
&-u_{\textcolor{Cerulean}{\text{to}}}+\sum\limits_{\text{x}\in\text{V}}u_{\text{x}}\text{Pr}(x\mid\textcolor{red}{\text{want}})
\end{align*}
Taking first derivative with respect to $u_{\textcolor{Cerulean}{\text{I}}}$ yields
\begin{align*}
\frac{\partial}{\partial u_{\textcolor{Cerulean}{\text{I}}}}J(\boldsymbol\theta)
=&-v_{\textcolor{red}{\text{want}}}[1-2\text{Pr}(\textcolor{Cerulean}{\text{I}}\mid\textcolor{red}{\text{want}})].
\end{align*}
Taking first derivative with respect to $u_{\textcolor{Cerulean}{\text{to}}}$ yields
\begin{align*}
\frac{\partial}{\partial u_{\textcolor{Cerulean}{\text{to}}}}J(\boldsymbol\theta)
=&-v_{\textcolor{red}{\text{want}}}[1-2\text{Pr}(\textcolor{Cerulean}{\text{to}}\mid\textcolor{red}{\text{want}})].
\end{align*}
So we will update corresponding parameters in the current window,
\begin{align*}
\begin{bmatrix}
v_{\textcolor{red}{\text{want}},1}\\\\
v_{\textcolor{red}{\text{want}},2}\\\\
u_{\textcolor{Cerulean}{\text{I}},1}\\\\
u_{\textcolor{Cerulean}{\text{I}},2}\\\\
u_{\textcolor{Cerulean}{\text{to}},1}\\\\
u_{\textcolor{Cerulean}{\text{to}},2}
\end{bmatrix}
&=
\begin{bmatrix}
v_{\textcolor{red}{\text{want}},1}\\\\
v_{\textcolor{red}{\text{want}},2}\\\\
u_{\textcolor{Cerulean}{\text{I}},1}\\\\
u_{\textcolor{Cerulean}{\text{I}},2}\\\\
u_{\textcolor{Cerulean}{\text{to}},1}\\\\
u_{\textcolor{Cerulean}{\text{to}},2}
\end{bmatrix}-\alpha
\begin{bmatrix}
\frac{\partial}{\partial v_{\textcolor{red}{\text{want}}}}J(\boldsymbol\theta)\\\\
\frac{\partial}{\partial v_{\textcolor{red}{\text{want}}}}J(\boldsymbol\theta)\\\\
\frac{\partial}{\partial u_{\textcolor{Cerulean}{\text{I}}}}J(\boldsymbol\theta)\\\\
\frac{\partial}{\partial u_{\textcolor{Cerulean}{\text{I}}}}J(\boldsymbol\theta)\\\\
\frac{\partial}{\partial u_{\textcolor{Cerulean}{\text{to}}}}J(\boldsymbol\theta)\\\\
\frac{\partial}{\partial u_{\textcolor{Cerulean}{\text{to}}}}J(\boldsymbol\theta)
\end{bmatrix}
\end{align*}
Similarly, we could follow these procedures until the end of corpus which completes one pass (also called one epoch). After many passes or epoches, it may converge to some extend and the average of $v_{\textcolor{red}{\text{center}}}$ and $u_{\textcolor{Cerulean}{\text{context}}}$ would be the learned representation for each word.

| Word     | Vector |
| ----------- | ----------- |
| I      | $[(u_{\textcolor{Cerulean}{\text{I}},1} +v_{\textcolor{red}{\text{I}},1})/2,(u_{\textcolor{Cerulean}{\text{I}},2} +v_{\textcolor{red}{\text{I}},2})/2]$|
| want      | $[(u_{\textcolor{Cerulean}{\text{want}},1} +v_{\textcolor{red}{\text{want}},1})/2,(u_{\textcolor{Cerulean}{\text{want}},2} +v_{\textcolor{red}{\text{want}},2})/2]$|
| to      |$[(u_{\textcolor{Cerulean}{\text{to}},1} +v_{\textcolor{red}{\text{to}},1})/2,(u_{\textcolor{Cerulean}{\text{to}},2} +v_{\textcolor{red}{\text{to}},2})/2]$|
| buy      |$[(u_{\textcolor{Cerulean}{\text{buy}},1} +v_{\textcolor{red}{\text{buy}},1})/2,(u_{\textcolor{Cerulean}{\text{buy}},2} +v_{\textcolor{red}{\text{buy}},2})/2]$|
| eat      |$[(u_{\textcolor{Cerulean}{\text{eat}},1} +v_{\textcolor{red}{\text{eat}},1})/2,(u_{\textcolor{Cerulean}{\text{eat}},2} +v_{\textcolor{red}{\text{eat}},2})/2]$|
| an      |$[(u_{\textcolor{Cerulean}{\text{an}},1} +v_{\textcolor{red}{\text{an}},1})/2,(u_{\textcolor{Cerulean}{\text{an}},2} +v_{\textcolor{red}{\text{an}},2})/2]$|
| Apple      |$[(u_{\textcolor{Cerulean}{\text{Apple}},1} +v_{\textcolor{red}{\text{Apple}},1})/2,(u_{\textcolor{Cerulean}{\text{Apple}},2} +v_{\textcolor{red}{\text{Apple}},2})/2]$|
| iPhone      |$[(u_{\textcolor{Cerulean}{\text{iPhone}},1} +v_{\textcolor{red}{\text{iPhone}},1})/2,(u_{\textcolor{Cerulean}{\text{iPhone}},2} +v_{\textcolor{red}{\text{iPhone}},2})/2]$|
| now      |$[(u_{\textcolor{Cerulean}{\text{now}},1} +v_{\textcolor{red}{\text{now}},1})/2,(u_{\textcolor{Cerulean}{\text{now}},2} +v_{\textcolor{red}{\text{now}},2})/2]$|


**What's Next?** 

- We could actually use matrix notations to represent everything we did above, which is easier to use in coding.
- The calculation of normalization factor $Z$ is a burden when billions of words punch you, so an alternative way, called negative sampling, could be used to avoid it.
- Implementation in Pytorch with another toy example should be fun! 