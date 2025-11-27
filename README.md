# 🌐 Learning in Diversity: Empowering Domain Generalized Semantic Segmentation with Style Injection

Official implementation of **“Learning in Diversity: Empowering Domain Generalized Semantic Segmentation with Style Injection”**, accepted to **IEEE Transactions on Multimedia (TMM)**, 2025.

---

## 📄 Paper

> **Learning in Diversity: Empowering Domain Generalized Semantic Segmentation with Style Injection**  
> Runtong Zhang, Fanman Meng, Haoran Wei,  Qingbo Wu, Linfeng Xu, Hongliang Li  
> IEEE Transactions on Multimedia, 2025  
> [[Paper]](https://arxiv.org/abs/xxxx.xxxxx)

---

## 📦 Environment Setup

```bash
conda create -n SID python=3.8 -y
conda activate SID
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install "mmsegmentation>=1.0.0"
pip install "mmdet>=3.0.0"
pip install xformers=='0.0.20'
pip install -r requirements.txt
pip install future tensorboard

