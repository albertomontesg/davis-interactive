# DAVIS Challenge 2018 Interactive Track

In this section we are going to explain in more detail how the Interactive Track of the DAVIS 2018 Challenge works. The technical challenges behind this track make us be cautious about this first edition, so we launch in beta mode. We will make our best to make it work, but we cannot guarantee it 100%.

## Dataset

The dataset used in this Interactive Challenge is made of video sequences with user annotated scribbles. The video sequences are the same as the DAVIS 2017 as well as the data split. The annotated scribbles have been performed by humans letting them decide which frame annotate. Once they knew which frame was the best to perform the annotations, they draw scribbles over the foreground objects appearing at the sequence.

For every sequence of the DAVIS 2017 dataset, there is available 3 different scribbles annotated for different users. On the next figure you can see an example of three different user annotations on the same sequence `dogs-jump`. Note that the annotations are performed on different frames as the user chose the more conveninet for him.

<div style="white-space: nowrap;">

<img src="/docs/images/scribbles/dogs-jump-scribble00.jpg" width="33.3%"/><img src="/docs/images/scribbles/dogs-jump-scribble01.jpg" width="33.3%"/><img src="/docs/images/scribbles/dogs-jump-scribble02.jpg" width="33.3%"/>

</div>

## Workflow

The aim of this challenge is to evaluate interactive models that can provide quality segmentation masks given scribbles annotations. We decided to use scribbles as it presents a more realistic scenario as the scribbles are more easy and fast to annotate than full segmentation masks.

The workflow to evaluate this interactive models is as follows. A set of samples is will be given to the user and this sample will be made of one video sequence in the dataset and one human annotated scribble at the beginning. 

!!! note
	As there are 3 annotation per each sequence, the same sequence will be evaluated multiple times in different samples.

With this starting scribble the user must run its model to perform a prediction of the masks for the whole sequence. The time taken to perform this prediction will be measured and used to evaluate the model. Once the user has made the prediction and submited to the framework, a new scribble for this sample will be returned simulating a real human interaction. This returned annotation will be an additional scribble performed on the frame where the prediction obtained worst Jaccard score. This annotations are performed by an automated robot which tries to simulate human annotations.

In the following images you can see an example about how the robot works. On the left there is a prediction performed by the user model on the worst frame, and on the right image, you can observe the generated scribbles by the robot. The robot focus on the zones where there has been an error in prediction and try to give a feedback as a human would do.

<div style="white-space: nowrap;">

<img src="/docs/images/workflow/pred_mask_overlay.jpg" width="50%"/><img src="/docs/images/workflow/generated_scribbles.jpg" width="50%"/>

</div>

The additional annotation given to the user should be used again by its model to perform a new prediction of the masks. This procedure will be repeated until a maximum number of interactions or until a timeout is hit. The timeout is proportional to the number of objects in the sequence.

!!! example
	If the maximum number of interactions is set to 8, and the timeout to perform this interaction is 240s, this gives us an average maximum time of 30 seconds per interaction. But as the timeout is proportional to the number of objects in the sequence, in the sequence `blackswan` the timeout will be 30s per interaction while for the `salsa` sequence will be 300s per interaction (it has 10 objects). This behaviour is to be fair with models that the prediction computation is proportional to the number of objects in the sequence.

### Local

This framework provides the possibility to evaluate locally your models. Locally can only be evaluated the `train` and `val` subsets and the evaluation results won't be used to rank at the challenge.

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

* $\mathcal{J}_{60s}$: Jaccard at 60 seconds. This metric will be computing the interpolation of the previous curve evaluating at 60 seconds. This will encourage the users to implement and test fast methods capable of giving good predictions in a short time.