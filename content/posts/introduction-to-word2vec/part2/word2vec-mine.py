import numpy as np

# our softmax function
def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()

# word2vec, Skip-Gram, model
class word2vec:
	def __init__(self):
		self.hidden_size 	= 2
		self.window_size 	= 1
		self.V 				= 9
		# learning rate, alpha
		self.learning_rate 	= 0.001

	def initial(self):
		self.W1 = np.random.uniform(-1, 1, (self.hidden, self.V))
		self.W2 = np.random.uniform(-1, 1, (self.V, self.hidden))

	def forward(self, x):
		self.center = np.dot(self.W1, x).reshape(self.hidden_size, 1)
		self.output = np.dot(self.W2, self.center)
		self.yhat = softmax(self.output).reshape(self.V, 1)
        return self.yhat
	
	def backward(self, y):
		# for W1
		dJ_dW1 = np.dot(self.W2, self.yhat - self.y)
		# for W2
		C = self.y.sum()
		dJ_dW2 = np.dot(C*self.yhat-self.y, self.center.T)
		# update
		self.W1 = self.W1 - self.learning_rate * dJ_dW1
		self.W2 = self.W2 - self.learning_rate * dJ_dW2

	def train(self, max_epochs):
        for epoch in range(1, max_epochs):       
            self.loss = 0
            for t in range(len(self.X_train)):
                x = self.X_train[t]
                y = self.Y_train[t]
                self.forward(x)
                self.backward(y)
                self.loss += -1*np.dot(self.y.T, np.log(self.yhat))
            print(f"epoch={epoch} with loss={self.loss}")
            self.learning_rate *= 1/( (1+self.learning_rate*epoch) )
             
    def predict(self,word,number_of_predictions):
        if word in self.words:
            index = self.word_index[word]
            X = [0 for i in range(self.V)]
            X[index] = 1
            prediction = self.feed_forward(X)
            output = {}
            for i in range(self.V):
                output[prediction[i][0]] = i
              
            top_context_words = []
            for k in sorted(output,reverse=True):
                top_context_words.append(self.words[output[k]])
                if(len(top_context_words)>=number_of_predictions):
                    break
      
            return top_context_words
        else:
            print("Word not found in dictionary")