# 2DseisvelGenerator
Author : Ankur Kumar <br><br>
This is a generative model that will generate the random  samples of high quality seismic velocity models of earth's subsurface. This is a good application of **Generative AI** in the field of Geoscience.
![image](https://github.com/user-attachments/assets/ef5b5e06-d513-4d25-a569-07d494787e79)
Go to my [Hugging Face](https://huggingface.co/kankur0007/2DseisvelGenerator) profile and learn how  you can generate high quality random samples of  seismic velocity models with few lines of code. You can aslo see my [Spaces](https://huggingface.co/spaces/kankur0007/2DseisvelGenerator) in Hugging Face Community. You can see more details about this generative model in **2DseisvelGenerator's model card**. So well , this is a quick introduction of my repositories. Now I am going to tell you that **"What are the fundamental principles working behined this model and why i choose this !** 🤗. So let's start

## 📃 A Brief Introduction to Seismic FWI and It's Challenges 🙇🏻
Seismic Full Waveform Inversion (FWI) is a sophisticated geophysical technique designed to produce high-resolution subsurface models by aligning observed seismic data with synthetic data generated from a predictive model. It involves iteratively updating a model of the Earth's properties (such as velocity, density, etc) until the synthetic waveforms match the observed ones. Unlike conventional inversion methods, which often rely on specific portions of seismic data like travel times or amplitude, FWI utilizes the entire seismic waveform, incorporating both amplitude and phase information. This comprehensive approach results in far more detailed subsurface models.There are two methods of approaching the seismic full waveform inversion problem first one is **traditional method** approach and another one is **data driven** approach. In traditional inversion approach FWI is difficult due to its **ill-posedness**, **nonlinearity**, **limited resolution** , **computational complexity** and **time-consuming nature**  [Learn More](https://www.redalyc.org/journal/465/46558134008/html/) [2]. In this approach our model may get stuck in **local minima**, results poor convergence.These complexities make FWI extreme difficult and time-intensive to solve. In contrast, **data-driven** approaches provide a powerful alternative to traditional FWI by leveraging deep learning to handle nonlinearity, reduce computational costs and enhance the resolution. Various neural network architectures have been explored, but challenges remain in generalizing to different source functions and ensuring robustness against noise and missing data.
#### What are those challenges ? Let's see an example 🌝 :
Our traditional neural networks and most standard machine learning models are based on the assumption that the data used for training and testing are independent and identically distributed (i.i.d.)[3]. Here independent means each data point is independent of the others and identically distributed means all data points are drawn from the same probability distribution. The i.i.d. assumption does not hold in many real-world scenarios due to unforeseen distributional shift , leading to challenges in OOD generalization[3].Full Waveform Inversion (FWI) faces significant challenges in generalizing to out-of-distribution (OOD) data, where the test data distribution differs from the training data distribution , causing degradation in models performance[3].This is just one example of the many challenges we face when trying to generalize Neural Networks for seismic Full Waveform Inversion (FWI) problems, we will see these latter.

#### 💁🏻‍♂️ Practical Solutions for Challenges in Data-Driven FWI : 
We can solve these problem by two common approach 
- **Build a large, diverse, and realistic dataset** : By applying data augmentation, macking data more diverse and random ...etc.
- **Build a network with strong generalization ability** : By applying Transfer Learning and Pretraining, Hybrid Models,Regularization and Physics Constraints,....etc<br><br>
In this article, I will only discuss the issues associated with synthetic data and explore strategies for making the data more diverse and random so that we can improve our model's performence.





## Waveforms Gone Wild: 🤖 AI-Powered Velocity Models to Improve Seismic FWI 
Recently, neural networks and deep learning techniques have become popular in FWI, enhancing its capability to solve complex inverse problems. However, most neural networks used in FWI are trained on synthetic data, because large-scale real world  datasets are not  available  due to issues such as high data acquisition costs, labeling costs, intellectual property concerns, or security concerns. Conducting seismic surveys in the real world is both time-consuming and expensive.  Gathering real seismic data requires fieldwork, sensor deployment, data acquisition, and processing, all of which are resource-intensive.Real-world seismic datasets are often limited in availability due to various constraints.  Access to proprietary datasets from industry or government sources may be restricted, and there may be a lack of sufficient data from certain geographical regions.  Moreover, real seismic data tends to be limited in quantity and may not adequately represent a wide range of geological scenarios. Synthetic data offers a solution to this challenge, providing an endless supply of data that can be tailored to the specific needs of the neural network model. **But there may be a problem ,synthetic data can be cursed with bias**, and various types of statistical biases can emerge in the generation and usage of synthetic data. These biases often result from assumptions, simplifications, or limitations in the data generation process. The biases can influence the performance of deep learning models, leading to  incorrect results when applied to real-world data. Here are some common types of statistical bias that can occur in synthetic data
##### 1. Selection Bias
This occurs when the generated synthetic data does not accurately represent the diversity of real-world data. It happens when certain scenarios, geological structures, or subsurface conditions are overrepresented or underrepresented in the synthetic dataset.  Selection bias can lead to models that perform well on the synthetic data but poorly on real-world data, as they may fail to generalize to unseen or underrepresented conditions.

##### 2. Simulated Data Bias
Synthetic data is typically generated based on predefined parameters (e.g., velocity, density, geological structures). If these parameters are chosen arbitrarily or without sufficient variation, the synthetic data may not cover the full spectrum of real-world scenarios.  Simulated data bias can cause models to learn patterns that are too specific to the synthetic environment, resulting in poor performance when exposed to more varied real-world data.

#### 3. Confirmation Bias 
This occurs when the synthetic data is generated in a way that confirms pre-existing hypotheses or assumptions about the subsurface structures. The data generation process may be unintentionally guided by the researchers' expectations, leading to datasets that align with their beliefs rather than objective reality.

#### 4. Sampling Bias 
Synthetic data is often generated through sampling predefined distributions (e.g., velocity values, wave frequencies). If the sampling process is biased—favoring certain ranges or distributions—the generated data will not represent the true variability in real seismic conditions.

#### 5. Noise Bias 
The noise added to synthetic data is often simplified or idealized compared to real seismic noise, which can be complex, correlated, and non-stationary. If the synthetic noise does not accurately represent the types of noise present in real data, it introduces a bias.<br><br>
So well , these are the some common statistical bias that may occure while during generation of synthetic data. [Learn more about bias](https://link.springer.com/article/10.1007/s11831-024-10134-2
). **Now the question arises "can we remove these inherent bias from dataset?", and the answer is No!**. But we can improve it to some extent, how?

### 💡 Idea : 
By diversifying  the dataset we can reduce bias from it to some extent. What i did , i took the 5000 random samples of each class (total 10 classes are present) from [Open FWI dataset](https://openfwi-lanl.github.io/docs/data.html)  and shuffled it randomly. After that i applied strong data augmentation like Flipping (Horizontal/Vertical), Rotation, Scaling, Cropping and Elastic Distortion to make my dataset more random and diverse. But still we have limited ammount of dataset and Hybrid Models like ***CNN-Transformer Fused Model*** needs large ammount of dataset to train.  <br><br>
Now what i want, to train a deep learning model that took the finite number of samples what i have and learns their distribution. After that we can generate infinite number of samples from that learned distribution. ***Generative models*** does this task very well. Before moving forward you must have clear idea about what are [Generative and Discriminative Models](https://www.baeldung.com/cs/ml-generative-vs-discriminative) and how [Deep Generative Models](https://youtu.be/Dmm4UG-6jxA?si=EiP7DCs6mjSb_3Pj) works.

### 🧸 Generative Models
Generative models try to capture the distribution of the training data, enabling them to generate new data samples from this learned distribution. There are so many types of diffusion models like GANs , VAE , Diffusion Model ,Markov Chain Models ... etc. 
![image](https://github.com/user-attachments/assets/e188a4e1-724c-474a-a03d-8c34e1675c04)

***Now for my task i am going to choose diffusion model, because this model has  shown exceptional performance in generating high-quality, diverse, and detailed images. Diffusion models provide better coverage of the data distribution compared to GANs and tend to produce more realistic outputs, especially in complex domains like natural images or seismic velocity models. [For evidence read this article](https://ar5iv.labs.arxiv.org/html/2105.05233)***

# Diffusion Models : 
***Diffusion models are inspired by non-equilibrium thermodynamics.*** They define a Markov chain of diffusion steps to slowly add random noise to data and then learn to reverse the ***diffusion process*** to construct desired data samples from the noise. Unlike VAE or flow models, diffusion models are learned with a fixed procedure and the latent variable has high dimensionality (same as the original data).
#### What is diffusion process ? 
A diffusion process is a type of stochastic process that models the continuous random movement of particles, prices, or other variables over time that follows the following characteristics
- **Continuous Path** : The process evolves continuously over time, with no sudden jumps.
- **Markov Property** : The future state of the process  $X_t$ depends only on its current state $X_{t-1}$ , not on the history of how it got there.

$$
\ P(X_t| X_{t-1},X_{t-2}.......X_{t-n}) = P(X_t | X_{t-1})
$$

- **Gaussian Increments**: The changes (increments) in the process over any small time interval are normally distributed with mean proportional to the time interval and variance proportional to both the time interval and a constant known as the diffusion coefficient. 

The diffusion process $X_t$ an often be described by a stochastic differential equation (SDE) of the form: 

$$
\ dX(t) = \mu (X(t),t)dt + \sigma (X(t),t)dW(t)
$$

#### Denoising Diffusion Probabilistic Models
[Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239) are a class of generative models that learn to model the distribution of data by gradually transforming a simple distribution (like Gaussian noise) into a complex data distribution. This transformation is done through a sequence of incremental denoising steps like forward diffusion , reverse diffusion and sampling.











### Acknowledgments : 
- [1] ***Deep Learning for Visual Data , University of California, Berkeley.*** <br>
- [2] . Zhu, M., Feng, S., Lin, Y., Lu, L., 2023. Fourier-DeepONet: Fourier-enhanced deep 
operator networks for full waveform inversion with improved accuracy, generalizability, and 
robustness.
- [3] . Liu, J., Shen, Z., He, Y., Zhang, X., Xu, R., Yu, H., Cui, P., 2023. ***Towards Out-Of-Distribution 
Generalization: A Survey***



