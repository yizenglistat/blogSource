---
date: 2022-02-17
lastmod: 2022-02-25
linktitle: 
title: GAN, Explained - Part 1
weight: 1
pined: true
draft: false
tags: ["generator", "review", "lstm"]
categories: ["ace seq2seq"]
<!-- next: /testing-c
prev: /testing-a -->
---

 In GAN, generator and discriminator work together in generation tasks. But do we really need the discriminator considering the existence of generation models like VAE.

<!--more-->

The **G**nerative **A**dversarial **N**etwork (GAN) is used to generate "realistic" new data given a training set. 

Let's start with an example. Suppose we have a vector $z$ sampled from a normal distribution as below.

$$z=\begin{bmatrix}
-0.3\\\\
0.4\\\\
-0.5
\end{bmatrix}$$

Our target for this example is to generate a long vector so that it could be reshaped into a digital image. 

