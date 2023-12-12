from simple_nn import NeuralNetwork

def mse_loss(Xt, X_hat):
  return sum([(x-y)**2 for x,y in zip(Xt, X_hat)])/len(Xt)

class LTC:
  def __init__(self):

    self.ode_steps = 5
    self.H = 1e-2 # step size

    self.hidden_nodes = 20
    self.nn = NeuralNetwork(input_size=2, hidden_size=self.hidden_nodes, output_size=1)
    self.f = lambda x : self.nn.forward([x])[0][0] # parameterized function

    self.tao = np.random.uniform(-1, 1)
    self.A = np.random.uniform(-1, 1)
  

  def forward(self, I, x_t):
    # ode solve
    for _ in range(self.ode_steps):
      y = self.f([I, x_t])
      x_t = (x_t + (self.H*y*self.A))/(1 + self.H*(1/self.tao + y))
    
    return x_t
  __call__ = forward

  def evaluate(self, Xt):
    x_t = Xt[0]
    h_hat = [x_t]
    for I in Xt:
      x_t = self(I, x_t)
      h_hat.append(x_t)

    return h_hat

  def fit(self, Xt):
    epochs = 250
    dp = 1e-6
    lr = 1e-2

    # forward pass
    h_hat = self.evaluate(Xt)

    # compute loss
    loss = mse_loss(Xt, h_hat)

    for epoch in range(epochs):
      # compute grads
      self.tao += dp
      loss_ = mse_loss(Xt, self.evaluate(Xt))
      d_tao = (loss_-loss)/dp
      self.tao -= dp

      self.A += dp
      loss_ = mse_loss(Xt, self.evaluate(Xt))
      d_A = (loss_-loss)/dp
      self.A -= dp

      d_W1, d_b1, d_W2, d_b2 = np.zeros((2, self.hidden_nodes)), np.zeros((1, self.hidden_nodes)), np.zeros((self.hidden_nodes, 1)), np.zeros((1, 1))

      # compute d_W1
      for i in range(d_W1.shape[0]):
        for j in range(d_W1.shape[1]):
          self.nn.W1[i][j] += dp
          loss_ = mse_loss(Xt, self.evaluate(Xt))
          d_W1[i][j] = (loss_-loss)/dp
          self.nn.W1[i][j] -= dp
      
      # compute d_W2
      for i in range(d_W2.shape[0]):
        for j in range(d_W2.shape[1]):
          self.nn.W2[i][j] += dp
          loss_ = mse_loss(Xt, self.evaluate(Xt))
          d_W2[i][j] = (loss_-loss)/dp
          self.nn.W2[i][j] -= dp

      # compute d_b1
      for i in range(d_b1.shape[0]):
        for j in range(d_b1.shape[1]):
          self.nn.b1[i][j] += dp
          loss_ = mse_loss(Xt, self.evaluate(Xt))
          d_b1[i][j] = (loss_-loss)/dp
          self.nn.b1[i][j] -= dp
      
      # compute d_b2
      for i in range(d_b2.shape[0]):
        for j in range(d_b2.shape[1]):
          self.nn.b2[i][j] += dp
          loss_ = mse_loss(Xt, self.evaluate(Xt))
          d_b2[i][j] = (loss_-loss)/dp
          self.nn.b2[i][j] -= dp

      # apply gradients
      self.nn.apply(d_W1, d_b1, d_W2, d_b2, lr=lr)
      self.tao -= lr*d_tao
      self.A -= lr*d_A

      # recompute
      h_hat = self.evaluate(Xt)
      loss = mse_loss(Xt, h_hat)

      if not epoch%50: print(f'epoch : {epoch}, loss : {loss}')

      #plt.plot(h_hat)
    
    #plt.plot(Xt)
