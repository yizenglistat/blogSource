---
date: 2022-02-28
lastmod: 2022-03-01
linktitle: Introduction to Word2vec Part 2 
title: Introduction to Word2vec - Part 2
pined: false
weight: 1
draft: false
tags: ["word2vec", "from scratch", "python"]
categories: ["ace word2vec"]
featured_image: /posts/introduction-to-word2vec/part2/word2vec-fig4.png
---

A tutorial, focused on Skip-Gram, for Word2vec with model architecture/implementation from scratch in python. 

<!--more-->

As we discussed in [Introduction to Word2vec - Part 1](/posts/introduction-to-word2vec/part1/), the word2vec (Skip-Gram) model is trying to learn a map from a word to a numerical representation (vector). Here, we focus more on modeling and implementations from scratch in python.

We will continue to employ notations and the example used in [Part 1](/posts/introduction-to-word2vec/part1/) and everything should be consistent;
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

**Model parameters.** In this model, only two weight matrices are our parameters, which gets together to be $\boldsymbol\theta$ in [Part 1](/posts/introduction-to-word2vec/part1));
- $W_{1}$ is the weight matrix connecting input layer and hidden layer. No bias terms. 

\begin{align*}
W_1^\top&=\begin{bmatrix}
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
=\begin{bmatrix}
v_{\textcolor{red}{\text{I}}}\\\\
v_{\textcolor{red}{\text{want}}}\\\\
v_{\textcolor{red}{\text{to}}}\\\\
v_{\textcolor{red}{\text{buy}}}\\\\
v_{\textcolor{red}{\text{eat}}}\\\\
v_{\textcolor{red}{\text{an}}}\\\\
v_{\textcolor{red}{\text{Apple}}}\\\\
v_{\textcolor{red}{\text{iPhone}}}\\\\
v_{\textcolor{red}{\text{now}}}\\\\
\end{bmatrix}
\end{align*}

- $W_{2}$ is the weight matrix connecting hidden layer and output layer (before softmax). No bias terms.

\begin{align*}
W_2&=\begin{bmatrix}
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
=\begin{bmatrix}
u_{\textcolor{Cerulean}{\text{I}}}\\\\
u_{\textcolor{Cerulean}{\text{want}}}\\\\
u_{\textcolor{Cerulean}{\text{to}}}\\\\
u_{\textcolor{Cerulean}{\text{buy}}}\\\\
u_{\textcolor{Cerulean}{\text{eat}}}\\\\
u_{\textcolor{Cerulean}{\text{an}}}\\\\
u_{\textcolor{Cerulean}{\text{Apple}}}\\\\
u_{\textcolor{Cerulean}{\text{iPhone}}}\\\\
u_{\textcolor{Cerulean}{\text{now}}}\\\\
\end{bmatrix}
\end{align*}

**Training data set.** Our ready-to-train data set should be in the following format:
- Input $\textcolor{red}{\text{center}}$ word $x$ is one-hot-encoder vector with size $|V|=9$. For example, *Apple* would be

\begin{align*}
x = \begin{bmatrix}
0\\\\
0\\\\
0\\\\
0\\\\
0\\\\
\textcolor{red}{1}\\\\
0\\\\
0
\end{bmatrix}
\end{align*}

- Target $y$ is the summation of one-hot-encoder $\textcolor{Cerulean}{\text{context}}$ vectors. For example, $\textcolor{Cerulean}{\text{context}}$ for *want* would be 

\begin{align*}
y = \begin{bmatrix}
\textcolor{Cerulean}{1}\\\\
0\\\\
\textcolor{Cerulean}{1}\\\\
0\\\\
0\\\\
0\\\\
0\\\\
0\\\\
0
\end{bmatrix}
\end{align*}

**Feedforward pass.** Given an input $x$, the model will ouput a $\hat{y}$ as follows

\begin{align*}
v_{\textcolor{red}{\text{center}}} 
=&
W_1x=
\begin{bmatrix}
v_{\textcolor{red}{\text{center}},1}\\\\
v_{\textcolor{red}{\text{center}},2}
\end{bmatrix}\\\\
W_2v_{\textcolor{red}{\text{center}}} 
=&
\begin{bmatrix}
u_{\textcolor{Cerulean}{\text{I}}}^\top v_{\textcolor{red}{\text{center}}}\\\\
u_{\textcolor{Cerulean}{\text{want}}}^\top v_{\textcolor{red}{\text{center}}}\\\\
u_{\textcolor{Cerulean}{\text{to}}}^\top v_{\textcolor{red}{\text{center}}}\\\\
u_{\textcolor{Cerulean}{\text{buy}}}^\top v_{\textcolor{red}{\text{center}}}\\\\
u_{\textcolor{Cerulean}{\text{eat}}}^\top v_{\textcolor{red}{\text{center}}}\\\\
u_{\textcolor{Cerulean}{\text{an}}}^\top v_{\textcolor{red}{\text{center}}}\\\\
u_{\textcolor{Cerulean}{\text{Apple}}}^\top v_{\textcolor{red}{\text{center}}}\\\\
u_{\textcolor{Cerulean}{\text{iPhone}}}^\top v_{\textcolor{red}{\text{center}}}\\\\
u_{\textcolor{Cerulean}{\text{now}}}^\top v_{\textcolor{red}{\text{center}}}
\end{bmatrix}\\\\
\hat{y}=&\text{softmax}(W_2v_{\textcolor{red}{\text{center}}})=\begin{bmatrix}
\frac{\exp(u_{\textcolor{Cerulean}{\text{I}}}^\top v_{\textcolor{red}{\text{center}}})}{Z}\\\\
\frac{\exp(u_{\textcolor{Cerulean}{\text{want}}}^\top v_{\textcolor{red}{\text{center}}})}{Z}\\\\
\frac{\exp(u_{\textcolor{Cerulean}{\text{to}}}^\top v_{\textcolor{red}{\text{center}}})}{Z}\\\\
\frac{\exp(u_{\textcolor{Cerulean}{\text{buy}}}^\top v_{\textcolor{red}{\text{center}}})}{Z}\\\\
\frac{\exp(u_{\textcolor{Cerulean}{\text{eat}}}^\top v_{\textcolor{red}{\text{center}}})}{Z}\\\\
\frac{\exp(u_{\textcolor{Cerulean}{\text{an}}}^\top v_{\textcolor{red}{\text{center}}})}{Z}\\\\
\frac{\exp(u_{\textcolor{Cerulean}{\text{Apple}}}^\top v_{\textcolor{red}{\text{center}}})}{Z}\\\\
\frac{\exp(u_{\textcolor{Cerulean}{\text{iPhone}}}^\top v_{\textcolor{red}{\text{center}}})}{Z}\\\\
\frac{\exp(u_{\textcolor{Cerulean}{\text{now}}}^\top v_{\textcolor{red}{\text{center}}})}{Z}
\end{bmatrix}=\begin{bmatrix}
\hat y_{\textcolor{Cerulean}{\text{I}}}\\\\
\hat y_{\textcolor{Cerulean}{\text{want}}}\\\\
\hat y_{\textcolor{Cerulean}{\text{to}}}\\\\
\hat y_{\textcolor{Cerulean}{\text{buy}}}\\\\
\hat y_{\textcolor{Cerulean}{\text{eat}}}\\\\
\hat y_{\textcolor{Cerulean}{\text{an}}}\\\\
\hat y_{\textcolor{Cerulean}{\text{Apple}}}\\\\
\hat y_{\textcolor{Cerulean}{\text{iPhone}}}\\\\
\hat y_{\textcolor{Cerulean}{\text{now}}}
\end{bmatrix}.
\end{align*}

Then we could find the most likely word in $V$ based on $\arg\max\hat{y}$ by find its corresponding maximum value in prediction. Herein $Z$ is same normalization factor defined in [Part 1](/posts/introduction-to-word2vec/part1).

**Loss function.** Minimize the cross-entropy is equivalent to maximize the log-likelihood function;

\begin{align*}
J
=& - \sum_{\textcolor{Cerulean}{\text{context}}\in V} y_{\textcolor{Cerulean}{\text{context}}}\log\hat{y}_{\textcolor{Cerulean}{\text{context}}}
\end{align*}

where $y\_{\textcolor{Cerulean}{\text{context}}}=1$ if $\textcolor{Cerulean}{\text{context}}$ exists in the $\textcolor{BurntOrange}{\text{window}}$ of current $\textcolor{red}{\text{center}}$, as indicated in true label $y$. And herein our $\hat y\_{\textcolor{Cerulean}{\text{context}}}=\text{Pr}(\textcolor{Cerulean}{\text{context}} \mid \textcolor{red}{\text{center}})$ defined in [Part 1](/posts/introduction-to-word2vec/part1).


**Gradient computations.** Taking the first derivative of $J$ with respect to $W_1$ will lead to the fact that only one $v_{\textcolor{red}{\text{center}}}$ is involved in the loss function. Thus

\begin{align*}
\frac{\partial J}{\partial W_1}
=& \begin{bmatrix}
0\\\\
\vdots\\\\
0\\\\
\frac{\partial J}{\partial v_{\textcolor{red}{\text{center}}}}\\\\
0\\\\
\vdots\\\\
0
\end{bmatrix}\\\\
\frac{\partial J}{\partial v_{\textcolor{red}{\text{center}}}}
=&-\sum_{\textcolor{Cerulean}{\text{context}}\in V}y_{\textcolor{Cerulean}{\text{context}}}u_{\textcolor{Cerulean}{\text{context}}}+\sum_{\textcolor{Cerulean}{\text{context}}\in V} u_{\textcolor{Cerulean}{\text{context}}}\hat y_{\textcolor{Cerulean}{\text{context}}}\\\\
=&W_2^\top(\hat y-y)\\\\
\frac{\partial J}{\partial W_1}=&W_2^\top(\hat y-y)x^\top
\end{align*}

Taking the first derivative of $J$ with respect to $W_2$ yields

\begin{align*}
\frac{\partial J}{\partial W_2}
=& \begin{bmatrix}
\frac{\partial J}{\partial u_{\textcolor{Cerulean}{\text{I}}}}\\\\
\frac{\partial J}{\partial u_{\textcolor{Cerulean}{\text{want}}}}\\\\
\frac{\partial J}{\partial u_{\textcolor{Cerulean}{\text{to}}}}\\\\
\frac{\partial J}{\partial u_{\textcolor{Cerulean}{\text{buy}}}}\\\\
\frac{\partial J}{\partial u_{\textcolor{Cerulean}{\text{eat}}}}\\\\
\frac{\partial J}{\partial u_{\textcolor{Cerulean}{\text{an}}}}\\\\
\frac{\partial J}{\partial u_{\textcolor{Cerulean}{\text{Apple}}}}\\\\
\frac{\partial J}{\partial u_{\textcolor{Cerulean}{\text{iPhone}}}}\\\\
\frac{\partial J}{\partial u_{\textcolor{Cerulean}{\text{now}}}}
\end{bmatrix}\\\\
\frac{\partial J}{\partial u_{\textcolor{Cerulean}{\text{context}}}}
=& -y_{\textcolor{Cerulean}{\text{context}}}v_{\textcolor{red}{\text{center}}}+Cv_{\textcolor{red}{\text{center}}}\hat y_{\textcolor{Cerulean}{\text{context}}}\\\\
\frac{\partial J}{\partial W_2}
=&-y+C\hat yv_{\textcolor{red}{\text{center}}}^\top=(C\hat y-y)v_{\textcolor{red}{\text{center}}}^\top
\end{align*}

where 

$$
C=\sum_{\textcolor{Cerulean}{\text{context}}\in V}y_{\textcolor{Cerulean}{\text{context}}}
$$

**Implementations in python.** Combining the feedforward and gradients, we are able to train the model given a dataset by using stochastic gradient desent.

Only trained for 100 epoches but it is kind of cool! You can play the code or notebook yourself: [word2vec-toy.py](https://github.com/yizenglistat/blogsource/blob/master/content/posts/introduction-to-word2vec/part2/word2vec-toy.py) and [word2vec-toy.ipynb](https://github.com/yizenglistat/blogsource/blob/master/content/posts/introduction-to-word2vec/part2/word2vec-toy.ipynb). 

```
import numpy as np

# our softmax function
def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()

# word2vec, Skip-Gram, model
class word2vec:
    def __init__(self):
        self.hidden_size = 2
        self.X_train = []
        self.y_train = []
        self.window_size = 1
        self.learning_rate = 0.001
        self.words = []
        self.word_index = {}
  
    def inital(self,V,data):
        self.V = V
        self.W1 = np.random.uniform(-0.8, 0.8, (self.hidden_size, self.V))
        self.W2 = np.random.uniform(-0.8, 0.8, (self.V, self.hidden_size))
          
        self.words = data
        for i in range(len(data)):
            self.word_index[data[i]] = i

    def forward(self, x):
        self.center = np.dot(self.W1, x).reshape(self.hidden_size, 1)
        self.output = np.dot(self.W2, self.center)
        self.yhat = softmax(self.output)
        return self.yhat

    def backward(self, x, y):
        # for W1
        dJ_dW1 = np.dot(np.dot(self.W2.T, self.yhat - y), x.T)
        # for W2
        C = y.sum()
        dJ_dW2 = np.dot(C*self.yhat - y, self.center.T)
        # update
        self.W1 = self.W1 - self.learning_rate * dJ_dW1
        self.W2 = self.W2 - self.learning_rate * dJ_dW2

    def train(self, max_epochs):
        for epoch in range(1, max_epochs):       
            self.loss = 0
            for t in range(len(self.X_train)):
                x = np.array(self.X_train)[t].reshape(self.V, 1)
                y = np.array(self.y_train)[t].reshape(self.V, 1)
                self.forward(x)
                self.backward(x, y)
                self.loss += -1*np.dot(y.T, np.log(self.yhat)).reshape(1,)[0]
            print(f"epoch={epoch} with loss={self.loss}")
            self.learning_rate *= 1/( (1+self.learning_rate*epoch))
```

```
epoch=1 with loss=55.09354880947808
epoch=2 with loss=55.064578930077715
epoch=3 with loss=55.035757463479
epoch=4 with loss=55.00711108927134
epoch=5 with loss=54.97866570194483
epoch=6 with loss=54.95044627223492
epoch=7 with loss=54.92247672005741
epoch=8 with loss=54.89477980056858
epoch=9 with loss=54.86737700457328
epoch=10 with loss=54.840288474171665
epoch=11 with loss=54.81353293419522
...
...
epoch=93 with loss=53.76422452733378
epoch=94 with loss=53.75955310729455
epoch=95 with loss=53.754964560776756
epoch=96 with loss=53.750456851387305
epoch=97 with loss=53.74602800353098
epoch=98 with loss=53.74167610040396
epoch=99 with loss=53.73739928205228
```

**What's Next?** 

- The calculation of normalization factor $Z$ is a burden when billions of words punch you, so an alternative way, called negative sampling, could be used to avoid it.  
- Implementation in Pytorch with another toy example should be fun!
- Word2vec could be aligned with GAN for textual generation tasks, i.e., seq2seq model or transformer.
