# Demo for Pytorch Implementaion of Data Driven and Physics Informed Deep O nets


This notebook gives a brief introduction to DeepOnets by Lu et al in their paper found [here](https://arxiv.org/abs/1910.03193). This is also part of my accompanying blog post [here](https://johncsu.github.io/DeepONet_Demo/)


Here we aim to create an Operator network that can solve the following similar 1D ODE problem given in the paper:

$$
\frac{d}{dx}F(x) = g(x) \\
F(0) = 0\\
for \  x \in [0,2]
$$

Where $g(x)$ is given. This is essentially integration of a 1D function with the initial condition $F(0) = 0$. In practice, the operator mapping can be fairly arbitary e.g. initial states, boundary conditions, partial data, etc.

We look at 2 different approaches: Data Driven and physics informed. Data Driven has faster convergence and is more straight forward but requires data which may not neccesarily be available for more complicated Operators. Physics informed does not need any data (other than $F(0) = 0$ and the equations governing the Operator) but is slower to train

Like regular deep learning tasks such as image classification or NLP, sampling too far out of training distribution (i.e. a $g(x)$ too wildly different from the given dataset) will cause the operator network to fail to produce the correct results.

<img src='Different a.gif'>

## Physics Informed Deep O Net (PINO)

We can also use physics informed training to train our Onet. In this example, we only use the derivative information and the initial condition $F(0)=0$ to train our network. No other data is used. For this example, we set the sampling point y to be the same as the points used to discretize $u(x)$ for convience.

This process takes longer and doesn't make too much sense for this example. But we can imagine problems where the sensors are sparse (e.g. only at the boundaries of domains) and therefore PINNs can be a good way to enforce laws at locations far from sensors.

To take derivatives efficiently, we use the `torch.func` library a jax like library designed for pytorch. We only want the derivatives of the output wrt to the sampling points y and not u(x). We use `vmap` to iterate over the 2 batch dimensions to achieve this and `jacrev` to calculate the derivatives.

Training takes quite a bit longer and is less stable and we alos need to start weighting losses to improve convergence. But we require significantly less data/information about the Operator.

<img src='PINO_vs_DataDriven.png'>

## Network

The network $G(u)(y)$ takes in two inputs: y and u. here 'u' represents the $g(x)$ in our problem. Because of the structure of networks, we have to discretize the function $g$ at a fixed number of grid points. We'll use 100 uniformly spread points across the range 0 to 2. Note that the location of each sensor $u(x_i)$ is implicitly given (i.e. we don't tell the network this info but it is baked via our training samples) and remains fixed for all $g) 

Here 'y' represents the points we want to query. Because $g(x)$ takes the range (0,2), any y points we specify will also need to take the range (0,2).

The Deep O Net uses a the stacked net from Lu et al. Essentially this is two seperate networks called the branch and trunk network that each handle the input function and sampling point respectively. They are merged at some latent representation through element wise multiplication and then passed through a final linear layer.

For both the trunk and branch net, we'll just use the standard MLP networks with tanh activations.

The number of sensors for $u(x)$ can be different to the number of points we sample for i.e. the input shapes to the trunk and branch net can be different. This means we need 2 'batch' dimensions to loop over - one to handle the trunk net and one for the branch net. Base on how `nn.Linear` works we define the inputs to the Onet as:

- branch net $u(x)$ has shape [B,1,M]
- trunk net input has shape [B,N,I]
- Output of Onet shape [B,N,O]

Where:
- B is the batch dimension for the branch net
- N is the 'batch' dimension for the trunk net
- M is the number of sensors that discretize u(x)/input dimension of the branch net
- I is the input dimension of the trunk net
- O is the output dimension size of the Onet

For this example:

$$
B = 10,000\\
N = 100\\
M = 100\\
I = 1\\
O = 1\\
$$

## Generating Data

We'll generate our derivatives $u(x)$ using Chebyshev polynomials. There's many different ways of generating random smooth functions but well use this way.

We use a mix of numpy and scipy integration to get the derivative and the integral function at each point. We then discretize both functions. for the derivative $u(x)$ we discretize using 100 uniformly spaced points. for $G(u)(y)$ we randomly sample $N = 100$ points in the domain

The tuple of data of the shape (y,u,Guy) where:
- y is the sampling points of shape $[10000,100,1]$
- u is the discretized derivative function of shape $[10000,1,100]$
- $G(u)(y)$ is target output of the Onet os shape $[10000,100,1]$

## Training Pipeline

We can then create a very straight forward training pipeline to train our data driven Onet. This is identical to other training pipelines such as image classification and we can simply use a dataloader


