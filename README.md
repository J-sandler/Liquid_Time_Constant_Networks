**Vanilla Liquid Time Constant Neural Network Implementation**

- This is a simple illustration of Liquid Time Constant Networks or LTC's, [Hasani,Lechner,Amini-2020] using a vanilla ode solver, 
and vanilla (BPTT) optimization. It is tested on a periodic (sine) function with 50 data points.

- The test can be run with sine_test.py which will instantiate and train a single LTC neuron to the test function.

- The dependencies for running this repo are simply numpy and matplotlib.

<br>
<br>
<br>

**Example Code:**

- LTC neurons operate as recurrent functions of input states and hidden states, whose dynamics
  are specified by a trainable but uniform differential equation.

- Computing a forward pass thus entails onfolding the differential eq, and at each ode step, computing
  the gradient (dx/dt) by taking a function of the input I, the hidden state x(t), time constant tao, and
  bias A.

- The ode step requires a delta t (H), set to 1e-2.
  
- The following code illustrates this process in the simplest fashion possible:

```
def forward(self, I, x_t):
    for step in range(ode_steps):
      y = F([I, x_t]) 
      x_t = (x_t + (H*y*A))/(1 + H*(1/tao + y))
    return x_t
```

<br>
<br>
<br>

**Results:**

- True Test Function [50 time steps]:
<img width="560" alt="Screen Shot 2023-12-12 at 12 07 39 AM" src="https://github.com/J-sandler/Liquid_Time_Constant_Networks/assets/108235294/e9aa2aa7-b95c-40f1-9b82-827c15b24e8f">

<br>
<br>
<br>

- LTC Neuron functions converging during training [moderately fitted]:
<img width="583" alt="Screen Shot 2023-12-12 at 12 06 58 AM" src="https://github.com/J-sandler/Liquid_Time_Constant_Networks/assets/108235294/e0e87adb-ac79-429b-babf-bbb41496da09">

<br>
<br>
<br>

- Final LTC Neuron fit [blue=sine, yellow=LTC]:
<img width="560" alt="Screen Shot 2023-12-12 at 12 07 53 AM" src="https://github.com/J-sandler/Liquid_Time_Constant_Networks/assets/108235294/b99581ce-b831-4cb6-86b3-d2bf10ed91a2">

<br>
<br>
<br>

*This was produced in direct reference to the paper Liquid Time-constant Networks - Ramin Hasani, Mathias Lechner, Alexander Amini, Daniela Rus, Radu Grosu : (MIT) (IST Austria) (TU Wien) Dec 14 2020*
