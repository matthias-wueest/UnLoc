
<h1 align="center">
  <ins>UnLoc: </ins><br>
  Leveraging Depth Uncertainties for<br> Floorplan Localization
</h1>


<p align="center">
  Matthias&nbsp;Wüest ·
  <a href="https://francisengelmann.github.io/">Francis&nbsp;Engelmann</a> ·
  <a href="http://miksik.co.uk/">Ondrej&nbsp;Miksik</a> · <br/>
  <a href="https://www.microsoft.com/en-us/research/people/mapoll/">Marc&nbsp;Pollefeys</a> ·
  <a href="https://cvg.ethz.ch/team/Dr-Daniel-Bela-Barath">Daniel&nbsp;Barath</a>
</p>


<p align="center">
  <img src="assets/iclr_logo.svg" width="160"/><br/>
</p>
<h3 align="center">
  <a href="https://www.arxiv.org/pdf/2509.11301">Paper</a> |
  <a href="#">Poster</a> |
  <a href="#citation">Citation</a>
</h3>


---

<p align="center">
  <img src="assets/teaser.png" alt="Teaser" width="95%">
  <br>
  <em>
    UnLoc predicts floorplan depth and uncertainty from an image sequence,
    generating a probability distribution over camera poses and outputting the most likely one.
  </em>
</p>

## Overview

We present **UnLoc**, an efficient data-driven solution for sequential camera localization within floorplans.
Unlike recent methods, UnLoc explicitly models the uncertainty in floorplan depth predictions and leverages off-the-shelf monocular depth networks pre-trained on large-scale datasets. Experimental results show substantial improvements in localization accuracy and robustness over existing state-of-the-art methods on multiple datasets.

<p align="center">
  <img src="assets/posterior.png" alt="Posterior evolution" width="95%">
  <br>
  <em>
    Posterior evolution showing how UnLoc achieves fast convergence to the true pose. 
  </em>
</p>

## News

- **[Feb 2026]** Code repository created
- **[Jan 2026]** Paper accepted to ICLR 2026

## Attribution

This code builds upon the [F³Loc](https://github.com/felix-ch/f3loc) framework by Chen et al. Our work introduces explicit uncertainty modeling in floorplan depth predictions and leverages pre-trained monocular depth networks, achieving substantial improvements in localization accuracy. We thank the authors for making their code publicly available.

## Installation
*Coming soon*

## Dataset
*Coming soon*

## Checkpoints
*Coming soon*

## Usage
*Coming soon*

## Results
*Coming soon*


## <a name="citation"></a>Citation
If you use this code, please cite both our work and the original F³Loc framework:

```bibtex
@InProceedings{wueest2026unloc,
  author    = {Wueest, Matthias and
               Engelmann, Francis and
               Miksik, Ondrej and
               Pollefeys, Marc and
               Barath, Daniel},
  title     = {UnLoc: Leveraging Depth Uncertainties for Floorplan Localization},
  booktitle = {Proc. ICLR},
  year      = {2026}
}

@InProceedings{chen2024f3loc,
  author    = {Chen, Changan and
               Wang, Rui and
               Vogel, Christoph and
               Pollefeys, Marc},
  title     = {F $\^{3}$ Loc: Fusion and Filtering for Floorplan Localization},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2024}
}
```


## License 
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.