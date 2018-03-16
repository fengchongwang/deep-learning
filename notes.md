## ConvNet
### Layer Size Calculation
#### Conv Layer
\begin{equation}
W_2 = \frac{W_1 - F + 2P}{S} + 1
\end{equation}
, where W_2 is the output width, W_1 being the input width, F being the filter's width, P being the amount of zero padding, S being the stride number.

When padding = 'same' and stride = 1,
\begin{equation}
P = \frac{F-1}{2}
\end{equation}
, and P is required to be an integer.

#### With Pooling
\begin{equation}
W_2 = \frac{W_1 - F}{S} + 1
\end{equation}
, where W_2 is the output width, W_1 being the width of the input width, F being the width of pooling layer, and S being stride number. 
Reference: 

* <https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/>
* <http://cs231n.github.io/convolutional-networks/#conv>

### Filter Weight Initialization  Method
\begin{equation}
w_i \sim N(0, \sqrt\frac{2}{n(1+a^2)})
\end{equation}
, where n is the number of input nodes for this layer, and a being the initialized alpha value for PReLu. For ReLu, it is 0.
Reference:

* He et. al. <https://arxiv.org/abs/1502.01852>

### Filter Bias Initialization
Initialize bias to be 0.

## Good Sources
* Kaggle winner's interview: <http://blog.kaggle.com/>