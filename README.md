# Adversarially Robust Control

This is the repo for paper ARC: Adversarially Robust Control Policies for Autonomous Vehicles.
The ARC framework trains control policies in an adversarial fashion for autonomous driving. 
The protagonist policy aims to drive safely, whilst the adversarial policy aims to cause collision.
The policies are trained end-to-end on the same loss, in a GAN-like fashion, within a RL training framework.
Multiple policies are used to create a more general and robust protagonist policy.

For further details see the paper: https://arxiv.org/abs/2107.04487


## Installation
Clone the repo

```bash
git clone https://github.com/sampo-kuutti/adversarially-robust-control
```

install requirements:
```bash
pip install -r requirements.txt
```

## Training the policies


To run ARC training run `train_arc.py`.

You can control the number of adversarial agents with the `--num_advs` argument.

## Citing the Repo

If you find the code useful in your research or wish to cite it, please use the following BibTeX entry.

```text
  title={ARC: Adversarially Robust Control Policies for Autonomous Vehicles},
  author={Kuutti, Sampo and Fallah, Saber and Bowden, Richard},
  booktitle={2021 IEEE International Intelligent Transportation Systems Conference (ITSC)},
  pages={522--529},
  year={2021},
  organization={IEEE}
}
```