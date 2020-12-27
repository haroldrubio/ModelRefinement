# Model Refinement
## A New Approach to Knowledge Distillation
I explore a new pipeline for knowledge distillation that builds off of the HuggingFace approach used to build a lighter version of the larger BERT model [1]. The hypothesis is that rich loss functions that allow gradient signals to directly propagate to each layer without being gated by intermediate layers enables for more effective weight representations. However, this is not to say that learning-at-scale with extremely deep and wide architectures is not needed. In fact, these networks are necessary starting points for refined models. Starting wide and deep allows networks to learn complex feature spaces at the per-layer level, where the shallow and refined networks would not have the capability of doing so, but could learn the same features at a cheaper cost when given the feature space at training time.
## Knowledge Distillation in the Computer Vision Domain
I first explore the transferability of the approach to the domain of computer vision using the landmark ResNet architecture [2].

# References
[1] Sanh, Victor, et al. “DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter.” ArXiv Preprint ArXiv:1910.01108, 2019.\
[2] He, Kaiming, et al. “Deep Residual Learning for Image Recognition.” 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770–778.
