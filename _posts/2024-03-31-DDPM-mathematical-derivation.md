---
layout: post
---
"Denoising Diffusion Probabilistic Models (DDPM)" presents a novel approach to generative modeling in the field of machine learning. Authored by Jonathan Ho et al., the paper introduces a framework that leverages diffusion processes to model complex data distributions. By iteratively denoising a sequence of noisy samples, DDPMs effectively learn the underlying data distribution, enabling high-quality generation of realistic images and other types of data. This innovative method has garnered attention for its ability to generate diverse and high-fidelity samples, making it a promising tool for various applications in image synthesis, data augmentation, and beyond. The diffusion model draws inspiration from non-equilibrium thermodynamics. Essentially, it borrows from non-equilibrium statistical physics. The main idea is to gradually break down the structure in a data distribution through a step-by-step forward diffusion process. Then, we develop a reverse diffusion process to bring back the structure in the data. In the DDPM paper, they define this process differently: during forward diffusion, an image is transformed into noise, and during reverse diffusion, this noise is turned back into the original image. In this process, an image is denoted as $$x_0$$, and this initial image is sampled from the data distribution, represented as $$q(x_0)$$. The graphical illustration of forward and reverse diffusion process of the DDPM paper is shown in Fig.(1).

![Alt text](/assets/images/ddpm.png "Figure 1: Reverse and forward diffusion process")

The joint probability distibution of $$x_1, x_2, \ldots, x_T$$ conditioned on $$x_0$$ is denoted as 
$$q(x_1,x_2,\ldots,x_T|x_0)$$. Based on Markov property we can write $$q(x_1,x_2,\ldots,x_T|x_0)$$ as follows,

$$q(x_1,x_2,\ldots,x_T|x_0) = \Pi_{t=1}^T q(x_t|x_{t-1}) \tag{1}$$

where 
$$q(x_t|x_{t-1})$$ is the transition kernel. In the DDPM paper the authors crafted the transition kernel as Gaussian perturbationn, and the mathematical expression is written as,

$$q(x_t|x_{t-1}) = {\cal N}\left(x_t; \sqrt{1-\beta_t}x_{t_-1}, \beta_t \mathbb{I} \right) \tag{2}$$

where $$\beta \in (0, 1)$$ is a hyperparameter chosen ahead of model training. Performing the operation described in equation (2) multiple times for $$t$$ steps takes a lot of time, especially when you need to compute 
$$q(x_t|x_0)$$. So, we aim to find a mathematical expression for $$q(x_t|x_0)$$ more efficiently. To do this, we'll employ a technique called the parameterization trick, as demonstrated below.

Considera random variable 
$$z$$ whic is sampled from a normal distribution as shown below,

 $$z \sim {\cal N}(\mu, \sigma^2) \tag{3a}$$

based on eqn.(3a) we can directly write the following equation,

$$z = \mu + \sigma \eta \tag{3b}$$

where $$\eta \in {\cal N}(0, 1)$$. The logic behind writing $$z$$ as in Eq.(3b) is very simple, if you calculate mean and variance of 
$$z$$ using Eq.(3b) it can be trivially calculated to be equal to 
$$E(z) = \mu$$ and $$Var(z) = \sigma^2$$. Using the same concept of reparameterization and applying to Eq.(2) we obtain the following,

$$x_t = \sqrt{1-\beta_t}x_t +\sqrt{\beta_t}\eta \tag{4}$$

where $$\eta \in {\cal N}(0, 1)$$. The eq.(4) can also be rewritten as follows,

$$x_t = \sqrt{1-\beta_t} \left( \sqrt{1-\beta_{t-1}} x_{t-2} + \sqrt{\beta_{t-1}}\eta\right) + \sqrt{\beta_t}\eta$$

$$x_t =  \sqrt{(1-\beta_t)(1-\beta_{t-1})} x_{t-2} + \sqrt{(1-\beta_t)\beta_{t-1}}\eta + \sqrt{\beta_t}\eta \tag{5a}$$

We denote 
$$\alpha_t = 1-\beta_t$$ and $$\bar{\alpha}_t = \Pi_{s=1}^t \alpha_s$$, and in terms $$\alpha_t$$ and $$\bar{\alpha}_t$$ rewrite Eq.(5a) as follows,

$$x_t = \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{\alpha_t(1-\alpha_{t-1})} \eta + \sqrt{1-\alpha_t}\eta \tag{5b}$$

If two random variable 
$$z_1$$ and $$z_2$$ such that $$z_1 \sim {\cal N}(\mu_1, \sigma^2_1)$$ and $$z_2 \sim {\cal N}(\mu_2, \sigma^2_2)$$. In that case $$z = z_1 + z_2$$ is random variable having the distribution 

$$z \sim {\cal N}(\mu_1+\mu_2, \sigma^2_1+\sigma^2_2)\tag{6}$$

Based on the concept defined in Eq.(6) we rewrite Eq.(5b) as follows,

$$x_t = \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + 0 + \sqrt{\alpha_t(1-\alpha_{t-1})} \eta + 0 + \sqrt{1-\alpha_t}\eta \tag{7}$$

where 
$$0 + \sqrt{\alpha_t(1-\alpha_{t-1})} \eta \sim {\cal N}(0, \alpha_t(1-\alpha_{t-1}))$$ and $$0 + \sqrt{1-\alpha_t}\eta \sim {\cal N}(0, 1-\alpha_t)$$. Using Eq.(6) one can write the following,

$$z^\prime \sim {\cal N}(0, 1-\alpha_t\alpha_{t-1}) \tag{8}$$

where 
$$z^\prime = 0 + \sqrt{\alpha_t(1-\alpha_{t-1})} \eta + 0 + \sqrt{1-\alpha_t}\eta$$. Using Eq.(8) in Eq.(7) we obtain the following,

$$x_t = \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{1-\alpha_t\alpha_{t-1}}\eta \tag{9}$$

Similar to Eq.(9), we can rewrite Eq.(4) in terms of $$\alpha$$'s.

$$x_t  = \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\eta \tag{10}$$

Based on Eq.(10) and Eq.(9) we can write the following,

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \eta \tag{11}$$

When $$\bar{\alpha}_T  \approx 0$$ then $$x_T$$is almost a normal distribution. Therefore, we can write,

$$q(x_T) = \int q(x_T|x_0) q(x_0) dx_0 = {\cal N}(x_T; 0, \mathbb{I}) \tag{12}$$

In simple terms, the forward process gradually adds noise to the data until all structures vanish. Equation (12) also highlights a key point: when we repeatedly apply the transition kernel 
$$q(x_t|x_{t-1})$$, as defined in Equation (2), to a sample $$x_0$$ from any distribution $$q(x_0)$$, we end up with a sample $$x_T$$ from a known distribution 
$${\cal N}(x_T; 0, \mathbb{I})$$.

However, this alone isn't very helpful. What's more useful is the reverse process, starting from isotropic Gaussian noise and transforming it into $$q(x_0)$$. But determining the distribution 
$$p(x_{t-1}\vert x_t)$$ is very challenging. To calculate this, we need to know the probability distribution of all the data, which can be quite complex. It has already been demonstrated in stochastic differential equations (like Brownian motion) by Langevin that the reverse diffusion process 
$$p(x_{t-1}\vert x_t)$$ can be approximated as a Gaussian distribution. Hence, we can parameterize the reverse diffusion process as follows,

$$p_\theta(x_{t-1}\vert x_t) = {\cal N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t,t)) \tag{13}$$

We aim to understand $$p_\theta (x_{t-1}\vert x_t)$$ in order to generate images that closely resemble our training dataset. For this purpose, we seek to maximize the log-likelihood of the observed data, as depicted below.

$$\underset{\theta}{\mathrm{max}}\log(p_\theta(x_0)) = \log\left( \int p(x_{0:T}) dx_{1:T}\right) \tag{14}$$

where $$p(x_{0:T})$$ is joint probability density.