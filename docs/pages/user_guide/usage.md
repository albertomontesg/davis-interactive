# Usage

The simplest usage can be as follows:

```python
from davisinteractive.session import DavisInteractiveSession

model = SegmentationModel() # Your segmentation model

# Configuration used in the challenges
max_nb_interactions = 8 # Maximum number of interactions 
max_time_per_interaction = 30 # Maximum time per interaction per object

# Total time available to interact with a sequence and an initial set of scribbles
max_time = max_nb_interactions * max_time_per_interaction # Maximum time per object

with DavisInteractiveSession(host='localhost', 
                             davis_root='path/to/DAVIS', 
                             max_nb_interactions=max_nb_interactions, 
                             max_time=max_time) as sess:
    while sess.next():
        # Get the current interaction's scribbles
        sequence, scribbles, _ = sess.get_scribbles()
        # Your model predicts the segmentation masks from the scribbles
        pred_masks = model(sequence, scribbles)
        # Submit your prediction
        sess.submit_masks(pred_masks)
        
	# Get the DataFrame report
    report = sess.get_report()
    
    # Get the global summary
    summary = sess.get_global_summary(save_file='summary.json')
```

Let us explain every component in detail to give a better understanding about how they work. 

## Session

A session is a sequence of samples where a sample is defined as a DAVIS sequence plus an initial set of scribbles annotated by a human.
Every sample is going to be evaluated interactively for a number of interactions (in a defined time window).
In order to be more realistic, we provide 3 manually annotated scribbles per sequence. Methods are evaluated starting from all 3 scribbles for each sequence, and the results are averaged.

The first step is to create a session to evaluate:

```python
with DavisInteractiveSession(host='localhost', 
                             davis_root='path/to/DAVIS', 
                             max_nb_interactions=max_nb_interactions, 
                             max_time=max_time) as sess:
```

This instructs the server where to perform the evaluation (`localhost` in the example), as well as the path of the DAVIS dataset files. 
When testing in the `train` and `val` subsets the evaluation should be done locally  (`host='localhost'`) and
parameters such as the `max_nb_interactions` per sample, the `max_time` per object, as well as the dataset split used, can be modified. 

!!! failure
    The remote evaluation server for the challenge is unavailable until the next edition.

The evalaution in the `test-dev` during the challange is performed remotely (`host='https://server.davischallenge.org'`), the `user_key` parameter should be set to the key sent in the registration email. For more information on how to obtain the user key and how to register to the challenge, please check [Challenge Section](/challenge/#remote). During the challenge, the `max_nb_interactions` and the `max_time` is set by the remote server so any value given to the `DavisInteractiveSession` class is ignored.

If you would like to not enforce a timeout and set only `max_nb_interactions`, you can set `max_time` to `None`. On the other hand, if you would like to not define the `max_nb_interactions`, you can set it to `None`. Then, you can keep interacting with the server until the timeout specified by `max_time` and proportional to the number of objects is reached for each sample. For more information about the DavisInteractiveSession class and its parameters please check [DavisInteractiveSession](/docs/session).

## Control Flow

In order to simplify the control flow for the user, the session object provides a function to move to the following interaction/sequence:

```python
while sess.next():
```

Once the timeout (`max_time` mutiplied by the number of objects in a sequence) or the maximum number of interactions (`max_nb_interactions`) is reached, this functions moves the evaluation to a new sequence or the same sequence with a different initial scribble. Otherwise, it provides more interactions for the current sequence.

## Obtain Scribbles

For a certain video sequence with an initial set of scribbles, there are multiple interactions (the number of interactions depends on the time limit or the maximum number of interactions per sample). In every interaction, the user has to call `get_scribbles` to obtain the scribbles for the next interaction. This returns a tuple with three elements:

* `sequence`: the name of the current sequence. This may be useful in case you are using a model that depends on the sequence of the DAVIS dataset which you are evaluating.
* `scribbles`: the scribbles of the current interaction. These scribbles by default are all the scribbles generated so far for the current sample (the first human annotated ones as well as all the ones automatically generated in the following interactions). If you call the method setting a flag `get_scribbles(only_last=True)` only the scribbles for the last interaction are returned.
* `new_sequence`: this is a flag indicating whether the given scribbles correspond to the first interaction of the sample.

```python
with DavisInteractiveSession(host='localhost', 
                             davis_root='path/to/DAVIS', 
                             max_nb_interactions=max_nb_interactions, 
                             max_time=max_time) as sess:
    while sess.next():
        sequence, scribbles, new_sequence = sess.get_scribbles(only_last)
        if new_sequence:
            model.load_weights(sequence)
        model.online_train(scribbles)
        pred_masks = model.predict()
        sess.submit_masks(pred_masks)
```

## Scribbles Transformations

The scribbles are represented as the different paths of the lines over each object ID. For more information about the scribbles format, check the page [Scribbles Format](scribbles).

This format may not be convenient for everybody, therefore we include some transformations in this framework:

* [`scribbles2mask`](/docs/utils.scribbles): it converts the paths of lines into a mask where the closest pixels of all the path points are set to the object ID of the line. This method also provides the possibility to apply the Bressenham's algorithm to fill in the path if two points of a line are sampled very distant in the mask.
* [`scribbles2points`](/docs/utils.scribbles): from the scribble, it extracts all the (x, y) coordinates of all the line points as well as its object ID.


## Prediction Submission

At the end of each interaction, the user must submit the masks predicted by his/her model to be evaluated in the server. As an optional parameter, the user may specify which frames have to be considered in order to compute the next scribbles. In order to do so, the parameter `next_scribble_frame_candidates` in the [submit_masks](/docs/session) function should be used. For example, `sess.submit_masks(pred_masks, [0, 1])` returns scribbles in the worst frame of the first two. By default, all the frames in a sequences are considered.

```python
pred_masks = model.predict()
sess.submit_masks(pred_masks)
```

## Final Result

Once the session has finished a report can be obtained using the `get_report` method. This method returns a Pandas DataFrame where every row contains the evaluation of every sequence, interaction and frame; as well as the timing of every interaction. From this report, information of the performance against processing time can be extracted for comparison among interactive methods.

For a global summary with the values and the evaluation curve, use the `get_global_summary` method. This method returns a dictionary with all the metrics and values used to evaluate and compare models. For more information about how the evaluation works, please go to the [Challenge Section](/challenge/#evaluation).
