# Usage

The simplest usage can be as follows:

```python
from davisinteractive.session import DavisInteractiveSession

model = SegmentationModel() # Your segmentation model

with DavisInteractiveSession(host='localhost', davis_root='path/to/davis') as sess:
    while sess.next():
        # Get the current iteration's scribbles
        sequence, scribbles, _ = sess.get_scribbles()
        # Your model predicts the segmentation masks from the scribbles
        pred_masks = model(sequence, scribbles)
        # Submit your prediction
        sess.submit_masks(pred_masks)
        
	# Get the result
    report = sess.get_report() 
```

Let us explain every component in detail to give a better understanding about how they work. 

## Session

A session is a sequence of samples (a DAVIS sequence plus an initial scribble annotated by a human).
Every sample is going to be evaluated interactively for a number of iterations (in a defined time window).
<!--- The whole evaluation will consist in the selected dataset of DAVIS sequences with all the annotated scribbles available. -->
In order to be more realistic, we provide 3 manually annotated scribbles per sequence. Submitted methods will be evaluated starting from all 3 scribbles for each sequence, and the results will be averaged.

The first step is to create a session to evaluate:

```python
with DavisInteractiveSession(host='localhost', davis_root='path/to/davis') as sess:
```

This instructs the server where to perform the evaluation (`localhost` or (soon) `remote`), as well as the path of the DAVIS dataset files. 
In case of development and local testing  (`host='localhost'` - the only option available for now), 
parameters such as the maximum number of interactions per object, the maximum interaction time per object, as well as the dataset split used, can be tuned. 

For more information about the class and its possible values please check [DavisInteractiveSession](/docs/session).

## Control Flow

In order to simplify the control flow for the user, the session provides a function to move to the following interaction/sequence:

```python
while sess.next():
```

Once the timeout or the maximum number of iterations is reached, this functions will move the evaluation to a new sequence or the same sequence with a different initial scribble. Otherwise, it will keep the current sequence in order to continue with more interactions.

## Obtain Scribbles

For every sample, there will be multiple iterations (depending on the time limit or the maximum number of iterations per sample). For every iteration you can call `get_scribbles` to obtain the scribbles for the next iteration. A tuple of three elements will be returned:

* `sequence`: the name of the sequence in the case you are using a model that depends on the sequence of the DAVIS dataset which you are evaluating.
* `scribbles`: the scribbles of the current iteration. This scribbles by default will be all the scribbles generated for the current sample (the first human annotated and all the automatic generated at next iterations). If you call the method setting a flag `get_scribbles(only_last=True)` only the last iteration's scribbles will be returned.
* `new_sequence`: this is a flag indicating whether the given scribbles correspond to the first iteration of the sample.

```python
with DavisInteractiveSession(host='localhost', davis_root='path/to/davis') as sess:
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

This format may not be convenient for everybody, and that is why some useful transformations are included on this framework:

* [`scribbles2mask`](/docs/utils.scribbles): it converts the paths of lines into a mask where the closest pixels of all the path points are set to the object ID of the line. This method also provides the possibility to apply the Bressenham's algorithm to fill in the path if two points of a line are sampled very distant on the mask.
* [`scribbles2points`](/docs/utils.scribbles): from the scribble, it extracts all the (x, y) coordinates of all the line points as well as its object ID.

If you think there is any new transformation or a modification to the current ones that might be useful to work with scribble data, please do not hesitate to send a [pull request](https://github.com/albertomontesg/davis-interactive/pulls).

## Prediction Submission

After each iteration, it is mandatory to submit the scribbles to evaluate and at the same time to automatically generate the scribble for the next iteration.

```python
pred_masks = model.predict()
sess.submit_masks(pred_masks)
```

## Final Result

Once the session has finished a report can be asked using the `get_report` method. This method returns a Pandas DataFrame where every row is the evaluation of every sequence, iteration and frame; as well as the timing of every iteration. From this report, information of the performance against processing time can be extracted for comparison among interactive methods.
