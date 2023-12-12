import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) 
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
        
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1 
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        y_pred = self.sigmoid(self.z2)
        
        return self.tanh(self.z2)*25 # scale output arbitrarily
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
      return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

    def apply(self, d_W1, d_b1, d_W2, d_b2, lr=1e-2):
      self.W1 -= lr*d_W1
      self.W2 -= lr*d_W2
      self.b1 -= lr*d_b1
      self.b2 -= lr*d_b2

if __name__ == '__main__':
  # Example Usage:
  nn = NeuralNetwork(input_size=2, hidden_size=3, output_size=2)
  X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
  y_pred = nn.forward(X)
  print(y_pred)
