# DAVIS Challenge 2018 Interactive Track

In this section we are going to explain in more detail how the Interactive Track of the DAVIS 2018 Challenge works. The technical challenges behind this track make us be cautious about this first edition, so we launch it in beta mode.

## Dataset

The Interactive Track is built on the DAVIS 2017 dataset. Video sequences, annotated objects, as well as data splits are the same as the ones in the Semi-Supervised track, and the sequences have been manually annotated with scribbles. The annotators were instructed to label all objects of a sequence in a representative frame (not necessarily the first frame of the sequence as in the Semi-Supervised track).

For every sequence of DAVIS 2017 there are 3 different sets of scribbles, annotated by different users. The figure below illustrates an example of three different user annotations on the same sequence `dogs-jump`. Note that annotations were performed on different frames, chosen by the respective users.

<div style="white-space: nowrap;">

<img src="/docs/images/scribbles/dogs-jump-scribble00.jpg" width="33.3%"/><img src="/docs/images/scribbles/dogs-jump-scribble01.jpg" width="33.3%"/><img src="/docs/images/scribbles/dogs-jump-scribble02.jpg" width="33.3%"/>

</div>

## Workflow

The aim of this challenge is to evaluate interactive models that can provide high quality segmentation masks, using scribbles and multiple interactions. Scribbles are a realistic form of supervision when it comes to video object segmentation, as they can be obtained much faster than full segmentation masks.

The workflow to evaluate these interactive models is as follows. A video sequence and a set of scribbles are given to the user. 

!!! note
	Since there are 3 annotations per each sequence, the same sequence will be evaluated multiple times, starting from different annotated scribbles.

The user must run their model to perform a prediction of the masks for the entire sequence, starting from the given scribble. As timing is important, the time taken to perform this prediction is measured. Once the user has made the prediction and submited their results, a new set of scribbles for this sequence will be returned, simulating human interaction. The returned annotation is an additional set of scribbles on the frame where the prediction failed the most (i.e. worst Jaccard score). These annotations are performed automatically, simulating human behaviour.

The following images show an example. The method makes a prediction, given the scribbles from the previous iteration(s) (left). Once the results are submitted, there are evaluated, and an additional set of scribbles is generated (right). The robot focuses on the zones where the error in prediction is the highest and tries to give feedback, as a human would do.

<div style="white-space: nowrap;">

<img src="/docs/images/workflow/pred_mask_overlay.jpg" width="50%"/><img src="/docs/images/workflow/generated_scribbles.jpg" width="50%"/>

</div>

The additional annotation given to the user should be used again by its model to perform a new prediction of the masks. This procedure will be repeated until a maximum number of interactions or a timeout is reached. The timeout is proportional to the number of objects in the sequence.

!!! example
	If the maximum number of interactions is set to 8, and the timeout to perform all interactions is 240s, this leads to a maximum time of 30 seconds per interaction. However, the timeout is proportional to the number of objects in the sequence. Thus the timeout for a sequence with a single object (eg. `blackswan`) will be 30s per interaction, while for a sequence with 10 objects (eg. `salsa`) the limit is set to 300s. This behaviour favours models for which the prediction time is proportional to the number of objects in the sequence.

### Local

This framework also provides the possibility to evaluate the methods locally. Local evaluation is possible only for the `train` and `val` subsets.

### Remote

In order to submit results to the Interactive Challenge, the frameworks allows to evaluate models agains a remote server. For remote evaluation, only the `test-dev` subset will be available and the results will be used to rank the user's models.

In order to run participate into the challenge a previous registration is required. To register, go to https://server.davischallenge.org and fill the form with your information. Then a mail will be sent to the provided email with a user key required to do the remote evaluation. This key should be put in your code in order to identify every user.

In addition, when the evaluation session is finished and a global sumary of the session is generated, a session ID will be given to the user. This session ID will allow the user identify its run and match them in the leaderboard.

## Evaluation

The main metric used to evaluate the predicted masks is the Jaccard similarity. Also the timing of the user's model when predicting the masks is taken into account. The evaluation samples will be the evaluation of the Jaccard for every sample and object of the dataset for each interaction. We are aware that some models may hit the timeout and don't reach the maximum number of interactions. In this case, for every sample with missing interactions, the evaluation of this missed interactions will be the same as the last interaction performed with 0 time cost.

With all this values, for every interaction the average jaccard and average timing will be computed and generate a curve of Jaccard vs Accumulated Time. On the following example you can see an example of how the curve can look like:

<div style="white-space: nowrap;">

<img src="/docs/images/workflow/evaluation.png" width="100%"/>

</div>

Given this curve, two parameters will be extracted to rank the user's models in order to compare them:

* $AUC$: Area under the curve. The area under the previous curve will be computed and normalized by the total time the model took to evaluate.

* $\mathcal{J}_{60s}$: Jaccard at 60 seconds. This metric will be computed performing a interpolation of the previous curve at 60 seconds. This will encourage the users to implement and test fast models capable of giving good predictions in a short time.
