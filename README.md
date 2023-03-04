# Control-Oriented Meta-Learning

This repository accompanies the papers:
- S. M. Richards, N. Azizan, J.-J. Slotine, and M. Pavone. ["Control-oriented meta-learning"](https://arxiv.org/abs/2204.06716). In *International Journal of Robotics Research (IJRR)*, 2023. In press.
- S. M. Richards, N. Azizan, J.-J. Slotine, and M. Pavone. ["Adaptive-control-oriented meta-learning for nonlinear systems"](https://arxiv.org/abs/2103.04490). In *Robotics: Science and Systems (RSS)*, 2021.

## Getting started

Ensure you are using Python 3. Clone this repository and install the packages listed in `requirements.txt`. In particular, this code uses [JAX](https://github.com/google/jax).


## Reproducing results

Training data can be generated with the commands `./generate pfar` and `./generate pvtol`.

Parameters can then be trained (for multiple training set sizes and random seeds) with the commands `./train pfar` and `./train pvtol`. This will take a while.

Test results can be reproduced with the commands `./test pfar` and `./test pvtol`. This may also take a while.

Finally, plots from the paper can be reproduced with commands `./plot pfar` and `./plot pvtol`.


## Citing this work

Please use the following BibTeX entries to cite this work.
```
@ARTICLE{RichardsAzizanEtAl2023,
author    = {Richards, S. M. and Azizan, N. and Slotine, J.-J. and Pavone, M.},
title     = {Control-oriented meta-learning},
year      = {2023},
journal   = {International Journal of Robotics Research},
url       = {https://arxiv.org/abs/2204.06716},
doi       = {10.48550/arXiv.2204.06716},
note      = {In press},
}

@INPROCEEDINGS{RichardsAzizanEtAl2021,
author    = {Richards, S. M. and Azizan, N. and Slotine, J.-J. and Pavone, M.},
title     = {Adaptive-control-oriented meta-learning for nonlinear systems},
year      = {2021},
booktitle = {Robotics: Science and Systems},
url       = {https://arxiv.org/abs/2103.04490},
doi       = {10.15607/RSS.2021.XVII.056},
}
```
