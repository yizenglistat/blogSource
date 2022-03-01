---
date: 2022-02-28
lastmod: 2022-03-01
linktitle: Introduction to Word2vec Part 2 
title: Introduction to Word2vec - Part 2
pined: true
weight: 1
prev: /
draft: false
tags: ["word2vec","neural network", "scratch-in-python"]
categories: ["ace word2vec"]
images: word2vec-fig1.png
---

A tutorial, focused on Skip-Gram, for Word2vec with model architecture/implementation from scratch in python. 

<!--more-->

As we discussed in [Introduction to Word2vec - Part 1](/posts/introduction-to-word2vec/part1/), the word2vec (Skip-Gram) model is trying to learn a map from words to a numerical representation (vector). Here, we focus more on modeling and implementations from scratch in python. 

We will continue to employ notations and the example used in [Part 1](/posts/introduction-to-word2vec/part1/) as follows;
- I want to buy an Apple iPhone.
- I want to eat an Apple now.

The vocabulary is denoted as $V=\\{$I, want, to, buy, eat, an, Apple, iPhone, now$\\}$. The model architecture is pictured in the following, where digits are hypothetical but the reasoning and logic should hold for any case.  

{{< image src="word2vec-fig4.png" alt="word2vec-fig4" >}}

**Model architecture.** We have one input layer, one hidden layer and one output layer. 
- Input layer is an one-hot-encoder for $\textcolor{red}{\text{center}}$ word with size $|V|=9$.
- Hidden layer has size 2 which is determined by us. Usually hundreds are valid.
- Output layer (after softmax) is an one-hot encoder for $\textcolor{Cerulean}{\text{context}}$ word with size $|V|=9$ within the $\textcolor{BurntOrange}{\text{window}}$. Recall the $\textcolor{BurntOrange}{\text{window}}$ size $m=1$ for our case.
- Note we do not have bias term (commonly used in neural network) here since we only consider dot product to account for similarity.
- Output layer (after softmax) will produce **same** prediction of all $\textcolor{Cerulean}{\text{context}}$ for current $\textcolor{red}{\text{center}}$ word.

**Model parameters.** In this model, only two weight matrices are our parameters;
- $W_{1}$ is the weight matrix connecting input layer and hidden layer. No bias terms. 
\begin{align*}
W_1&=\begin{bmatrix}
v_{\textcolor{red}{\text{I}},1},v_{\textcolor{red}{\text{I}},2}\\\\
v_{\textcolor{red}{\text{want}},1},v_{\textcolor{red}{\text{want}},2}\\\\
v_{\textcolor{red}{\text{to}},1},v_{\textcolor{red}{\text{to}},2}\\\\
v_{\textcolor{red}{\text{buy}},1},v_{\textcolor{red}{\text{buy}},2}\\\\
v_{\textcolor{red}{\text{eat}},1},v_{\textcolor{red}{\text{eat}},2}\\\\
v_{\textcolor{red}{\text{an}},1},v_{\textcolor{red}{\text{an}},2}\\\\
v_{\textcolor{red}{\text{Apple}},1},v_{\textcolor{red}{\text{Apple}},2}\\\\
v_{\textcolor{red}{\text{iPhone}},1},v_{\textcolor{red}{\text{iPhone}},2}\\\\
v_{\textcolor{red}{\text{now}},1},v_{\textcolor{red}{\text{now}},2}\\\\
\end{bmatrix}
\end{align*}
- $W_{2}$ is the weight matrix connecting hidden layer and output layer (before softmax). No bias terms.
\begin{align*}
W_2^\top&=\begin{bmatrix}
u_{\textcolor{Cerulean}{\text{I}},1},u_{\textcolor{Cerulean}{\text{I}},2}\\\\
u_{\textcolor{Cerulean}{\text{want}},1},u_{\textcolor{Cerulean}{\text{want}},2}\\\\
u_{\textcolor{Cerulean}{\text{to}},1},u_{\textcolor{Cerulean}{\text{to}},2}\\\\
u_{\textcolor{Cerulean}{\text{buy}},1},u_{\textcolor{Cerulean}{\text{buy}},2}\\\\
u_{\textcolor{Cerulean}{\text{eat}},1},u_{\textcolor{Cerulean}{\text{eat}},2}\\\\
u_{\textcolor{Cerulean}{\text{an}},1},u_{\textcolor{Cerulean}{\text{an}},2}\\\\
u_{\textcolor{Cerulean}{\text{Apple}},1},u_{\textcolor{Cerulean}{\text{Apple}},2}\\\\
u_{\textcolor{Cerulean}{\text{iPhone}},1},u_{\textcolor{Cerulean}{\text{iPhone}},2}\\\\
u_{\textcolor{Cerulean}{\text{now}},1},u_{\textcolor{Cerulean}{\text{now}},2}\\\\
\end{bmatrix}
\end{align*}

**Feedforward pass.** TO BE CONTINUE

**Gradient computations.** TO BE CONTINUE

**Training data set.** TO BE CONTINUE

**Implementations in python.** TO BE CONTINUE

