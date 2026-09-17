# Challenge: What Replaced Optical Flow?

The two-stream model computes optical flow explicitly, because motion is hard for a network to learn implicitly from raw frames. Ten years on — do modern video models still bother computing optical flow at all? If not, how do they capture motion instead?

## Work Through This

1. Pick **one** modern video model — I3D, SlowFast, a video transformer (TimeSformer), or VideoMAE — and find out **how it represents motion**. Does it use optical flow at all, or something else entirely?
2. What's the trade-off? What does the newer approach lose, and what does it gain, compared to explicit optical flow?

---

## Resources

Pick one model below as your starting point — each links to the original paper (and code, where available).

| Model | Paper |
|---|---|
| **I3D** (Inflated 3D ConvNets) | [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750) — Carreira & Zisserman, 2017 |
| **SlowFast** | [SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982) — Feichtenhofer et al., 2019 |
| **TimeSformer** | [Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/abs/2102.05095) — Bertasius, Wang & Torresani, 2021 |(https://github.com/facebookresearch/TimeSformer) |
| **VideoMAE** | [VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://arxiv.org/abs/2203.12602) — Tong et al., 2022 | 

Compare whichever one you pick against the two-stream paper's explicit optical-flow stream: what does the newer method trade away (interpretability, guaranteed motion signal) for what it gains (end-to-end learning, speed, less preprocessing)?
