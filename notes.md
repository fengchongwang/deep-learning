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

### Hyperparameter Selection
* use ELU non-linearity without batchnorm or ReLU with it.
* apply a learned colorspace transformation of RGB.
* use the linear learning rate decay policy.
* use a sum of the average and max pooling layers.
* use mini-batch size around 128 or 256. If this is too big for your GPU,
decrease the learning rate proportionally to the batch size.
* use fully-connected layers as convolutional and average the predictions for
the final decision.
* when investing in increasing training set size, check if a plateau has not
been reach.
* cleanliness of the data is more important then the size.
* if you cannot increase the input image size, reduce the stride in the consequent
layers, it has roughly the same effect.
* if your network has a complex and highly optimized architecture, like e.g.
GoogLeNet, be careful with modifications.

Reference:

* Systematic evaluation of CNN advances on the ImageNet by Dmytro Mishkin, Nikolay Sergievskiy, Jiri Matas, 2016 (<https://arxiv.org/abs/1606.02228>)

## Good Sources
* Kaggle winner's interview: <http://blog.kaggle.com/>