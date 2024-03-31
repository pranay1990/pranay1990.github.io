---
layout: post
---
"Denoising Diffusion Probabilistic Models (DDPM)" presents a novel approach to generative modeling in the field of machine learning. Authored by Jonathan Ho et al., the paper introduces a framework that leverages diffusion processes to model complex data distributions. By iteratively denoising a sequence of noisy samples, DDPMs effectively learn the underlying data distribution, enabling high-quality generation of realistic images and other types of data. This innovative method has garnered attention for its ability to generate diverse and high-fidelity samples, making it a promising tool for various applications in image synthesis, data augmentation, and beyond. The diffusion model draws inspiration from non-equilibrium thermodynamics. Essentially, it borrows from non-equilibrium statistical physics. The main idea is to gradually break down the structure in a data distribution through a step-by-step forward diffusion process. Then, we develop a reverse diffusion process to bring back the structure in the data. In the DDPM paper, they define this process differently: during forward diffusion, an image is transformed into noise, and during reverse diffusion, this noise is turned back into the original image. In this process, an image is denoted as $$x_0$$, and this initial image is sampled from the data distribution, represented as $$q(x_0)$$. The graphical illustration of forward and reverse diffusion process of the DDPM paper is shown in Fig.(1).

$$\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)\tag{1}$$


Equation is the famous mass-energy equivalence equation $$\sqrt{\$4}$$. Here I am.


Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: http://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
