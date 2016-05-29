Conditional Image Generation Using Deep Convolutional Generative Adversarial Networks

Based on "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks", by Radford et al. ([Paper] (http://arxiv.org/pdf/1511.06434v2.pdf)). All credit for original code goes to:  https://github.com/soumith/dcgan.torch. 

The paper, poster, and presentation slides for this work can be found here ([Paper](https://www.dropbox.com/s/maq72ci799bez2o/written_final_report.pdf?dl=0), [Poster](https://www.dropbox.com/s/wr90jnfs5vpzvhv/poster.pdf?dl=0), [Slides](https://www.dropbox.com/s/icsxtn1596lh3ou/oral_presentation.pdf?dl=0)). I do not provide an extensive overview of my code in this README as it is not production ready nor complete. However the main changes from the original torch code are in main_conditioned.lua and generate_conditioned.lua. main_conditioned.lua contains the code to train the model, while generate_conditioned.lua contains the code to generate images based on a trained model. There are also various lua and python scripts in the data/ subdirectory which deal with the loading and preprocessing of training data.

Special thanks to Fisher Yu and Prof. Jianxiong Xiao for advising with this work. The code was run on the Princeton VisionGPU cluster. 
