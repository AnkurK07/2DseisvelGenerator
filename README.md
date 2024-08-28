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












### Acknowledgments : 
- [1] ***Deep Learning for Visual Data , University of California, Berkeley.*** <br>
- [2] . Zhu, M., Feng, S., Lin, Y., Lu, L., 2023. Fourier-DeepONet: Fourier-enhanced deep 
operator networks for full waveform inversion with improved accuracy, generalizability, and 
robustness.
- [3] . Liu, J., Shen, Z., He, Y., Zhang, X., Xu, R., Yu, H., Cui, P., 2023. ***Towards Out-Of-Distribution 
Generalization: A Survey***



