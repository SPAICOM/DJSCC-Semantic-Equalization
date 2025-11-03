# Semantic Channel Equalization Strategies for Deep Joint Source-Channel Coding

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2510.04674-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2510.04674)
[![License](https://img.shields.io/badge/Code%20License-MIT-yellow)](https://github.com/PKU-YuanGroup/LanguageBind/blob/main/LICENSE)

 <br>

</h5>


> [!TIP]
> Deep joint source-channel coding (DeepJSCC) has emerged as a powerful paradigm for end-to-end semantic communications, jointly learning to compress and protect task-relevant features over noisy channels. However, existing DeepJSCC schemes assume a shared latent space at transmitter (TX) and receiver (RX) - an assumption that fails in multi-vendor deployments where encoders and decoders cannot be co-trained. This mismatch introduces "semantic noise", degrading reconstruction quality and downstream task performance. In this paper, we systematize and evaluate methods for semantic channel equalization for DeepJSCC, introducing an additional processing stage that aligns heterogeneous latent spaces under both physical and semantic impairments. We investigate three classes of aligners: (i) linear maps, which admit closed-form solutions; (ii) lightweight neural networks, offering greater expressiveness; and (iii) a Parseval-frame equalizer, which operates in zero-shot mode without the need for training. Through extensive experiments on image reconstruction over AWGN and fading channels, we quantify trade-offs among complexity, data efficiency, and fidelity, providing guidelines for deploying DeepJSCC in heterogeneous AI-native wireless networks.


## ️ Attribution

This repository is based on the original work by [Lorenzo Pannacci](https://www.linkedin.com/in/lorenzo-pannacci/).
You can find the original repository here:  
 [https://github.com/LorenzoPannacci/DJSCC-Semantic-Equalization](https://github.com/LorenzoPannacci/DJSCC-Semantic-Equalization)

## Dependencies  

### Using `pip` package manager  

It is highly recommended to create a Python virtual environment before installing dependencies. In a terminal, navigate to the root folder and run:  

```bash
python -m venv <venv_name>
```

Activate the environment:  

- On macOS/Linux:  

```bash
source <venv_name>/bin/activate
```

- On Windows:  

```bash
<venv_name>\Scripts\activate
```

Once the virtual environment is active, install the dependencies:  

```bash
pip install -r requirements.txt
```

You're ready to go!   

### Using `uv` package manager (Highly Recommended)  

[`uv`](https://github.com/astral-sh/uv) is a modern Python package manager that is significantly faster than `pip`.  

#### Install `uv`  

To install `uv`, follow the instructions from the [official installation guide](https://github.com/astral-sh/uv#installation).  

#### Set up the environment and install dependencies  

Run the following command in the root folder:  

```bash
uv sync
```

This will automatically create a virtual environment (if none exists) and install all dependencies.  

You're ready to go!   

## Authors

- [Lorenzo Pannacci](https://www.linkedin.com/in/lorenzo-pannacci/)
- [Simone Fiorellino](https://scholar.google.com/citations?hl=en&user=nKMc4GQAAAAJ)
- [Mario Edoardo Pandolfo](https://scholar.google.com/citations?user=wAeScL8AAAAJ&hl)
- [Emilio Calvanese Strinati](https://scholar.google.com/citations?user=bWndGhQAAAAJ)
- [Paolo Di Lorenzo](https://scholar.google.com/citations?hl=en&user=VZYvspQAAAAJ)

## Used Technologies

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
