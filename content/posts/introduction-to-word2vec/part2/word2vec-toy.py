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


# clean data and format data

import string

def process_data(corpus):
    training_data = []
    sentences = corpus.split(".")
    for i in range(len(sentences)):
        sentences[i] = sentences[i].strip()
        sentence = sentences[i].split()
        x = [word.lower() for word in sentence]
        training_data.append(x)
    return training_data

def prepare_data(sentences, w2v):
    
    data = {}
    for sentence in sentences:
        for word in sentence:
            if word not in data:
                data[word] = 1
            else:
                data[word] += 1
    V = len(data)
    data = sorted(list(data.keys()))
    vocab = {}
    for i in range(len(data)):
        vocab[data[i]] = i
    
    for sentence in sentences:
        for i in range(len(sentence)):
            center_word = [0 for x in range(V)]
            center_word[vocab[sentence[i]]] = 1
            context = [0 for x in range(V)]
            for j in range(i-w2v.window_size,i+w2v.window_size+1):
                if i!=j and j>=0 and j<len(sentence):
                    context[vocab[sentence[j]]] += 1
            w2v.X_train.append(center_word)
            w2v.y_train.append(context)
    
    w2v.inital(V,data)
  
    return vocab



corpus = "I want to buy an Apple iPhone. I want to eat an Apple now."
max_epochs = 100

training_data = process_data(corpus)
w2v = word2vec()
_ = prepare_data(training_data, w2v)
w2v.train(max_epochs=max_epochs)