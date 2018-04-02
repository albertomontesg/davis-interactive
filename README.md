# DAVIS Interactive Evaluation Framework


[![Travis](https://img.shields.io/travis/albertomontesg/davis-interactive.svg?style=for-the-badge)](https://travis-ci.org/albertomontesg/davis-interactive) [![Codecov branch](https://img.shields.io/codecov/c/github/albertomontesg/davis-interactive/master.svg?style=for-the-badge)](https://codecov.io/gh/albertomontesg/davis-interactive) [![license](https://img.shields.io/github/license/mashape/apistatus.svg?style=for-the-badge)](https://github.com/albertomontesg/davis-interactive/blob/master/LICENSE)

This is a framework to evaluate interactive segmentation models over the [DAVIS](http://davischallenge.org/index.html) dataset. The code aims to provide an easy-to-use interface to test and validate interactive segmentation models.

This is the tool that will be used to evaluate the DAVIS Challenge on Video Object Segmentation 2018 on the interactive track. More info about the challenge on the [website](http://davischallenge.org/challenge2018/interactive.html).

**Note**: code still under development.

## DAVIS Scribbles

On previous DAVIS Challenge the task consisted on object segmentation in a semisupervised manner. The input given was the ground truth mask of the first frame. For DAVIS interactive challenge we change the annotation to scribbles which can be annotated faster by humans.

<img src="docs/images/scribbles/dogs-jump-image.jpg" width="240"/> <img src="docs/images/scribbles/dogs-jump-scribble01.jpg" width="240"/> <img src="docs/images/scribbles/dogs-jump-scribble02.jpg" width="240"/>

The interactive annotation and segmentation consist on a iterative loop which is going to be evaluated as follows:

* On the first iteration, a human annotated scribble will be provided to the segmentation model. All the scribbles are annotated over the DAVIS dataset and the objects annotated will be the same as the ground truth masks. **Note**: the annotated frame can be any of the sequence as the humans where asked to annotate the frames that found most relevant and meaningfull to annotate.
* During the rest of the iterations, once the predicted masks have been submitted, an automated scribble is generated simulating human annotation. The new annotation will be performed on a single frame and this frame will be chosen as the worst on the evaluation metric.

**Evaluation**: For now, the evaluation metric will be the Jaccard similarity $\mathcal{J}$.

## Citation

Please cite both papers in your publications if DAVIS or this code helps your research.

```tex
@article{Caelles_arXiv_2018,
  author = {Sergi Caelles and Alberto Montes and Kevis-Kokitsi Maninis and Yuhua Chen and Luc {Van Gool} and Federico Perazzi and Jordi Pont-Tuset},
  title = {The 2018 DAVIS Challenge on Video Object Segmentation},
  journal = {arXiv:1803.00557},
  year = {2018}
}
```

```latex
@inproceedings{Perazzi2016,
  author = {F. Perazzi and J. Pont-Tuset and B. McWilliams and L. {Van Gool} and M. Gross and A. Sorkine-Hornung},
  title = {A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation},
  booktitle = {Computer Vision and Pattern Recognition},
  year = {2016}
}
```

