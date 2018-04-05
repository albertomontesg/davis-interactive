# Usage

The design idea behind this framework is to have an user API as easy as possible and flexible enought to support different user cases.

The simplest usage can be the following:

```python
from davisinteractive.session import DavisInteractiveSession

model = SegmentationModel() # Your segmentation model

with DavisInteractiveSession(host='localhost', davis_root='path/to/davis') as sess:
    while sess.next():
        # Get the current iteration scribbles
        sequence, scribbles, _ = sess.get_scribbles()
        # Your model predicts the segmentation masks from the scribbles
        pred_masks = model(sequence, scribbles)
        # Submit your prediction
        sess.submit_masks(pred_masks)
        
	# Get the result
    report = sess.get_report() 
```

Now lets explain every component in detail to give a better understanding about how this works. 

## Session

The first step is to create a session to evaluate:

```python
with DavisInteractiveSession(host='localhost', davis_root='path/to/davis') as sess:
```

This can specify the server where to perform the evaluation against, as well as the path of the DAVIS dataset files. In case of development and local testing against `localhost` (the only option available for now), the maximum number of interations per sample or the time per sample when evaluating can be customize as well as the dataset of the evaluation. 

For more information about the class and its possible values please check [DavisInteractiveSession](/docs/session).

## Control Flow

To make the framework as eassy as possible and don't leave the control flow of the evaluation to the user, the session gives a method to know for how long the session is running:

```python
while sess.next():
```

Also this function is necessary to be called after each iteration as it is responsible to move the evaluation to the next iteration or sample if maximum time or maximum number of iterations have been hit.

## Obtain Scribbles

For every sample there will be multiple iterations (depending on the time limit or the maximum number of iterations per sample). For every iteration you can call `get_scribbles` to obtain the scribbles for the next iteration. A tuple of three elements will be returned:

* `sequence`: the name of the sequence in the case you are using a model that depends on the sequence of the DAVIS dataset which you are evaluating.
* `scribbles`: the scribbles of the current iteration. This scribbles by default will be all the scribbles generated for the current sample (the first human annotated and all the automatic generated at next iterations). If you call the the method setting a flag `get_scribbles(only_last=True)` only the last iteration scribbles will be returned.
* `new_sequence`: this is a flag indicating if the given scribbles correspond to the first iteration of the sample. This might be useful in the case a model is being online trained at every iteration and at sample change need a reset. Example:

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

The scribbles are returned in a format which are represented the path of the differents lines annotated with its object IDs. For more information about the scribbles format, check the page [Scribbles Format](scribbles).

This format may not be convenient for everybody, and that's why some useful transformations are included on this framework:

* [`scribbles2mask`](/docs/utils.scribbles): converts the paths of lines into a mask where the closest pixels of all the paths points are set to the object id of the line. This method also provides the possibility to apply the Bresenham algorithm to fill the path if two points of a line are sampled very distant on the mask.
* [`scribbles2points`](/docs/utils.scribbles): from the scribble extracts all the (x, y) coordinates of all the line points as well as its object ID.

If you think there is any new transformation or a modification to the currents ones that might be useful to work with scribbles data, please do not hesitate to send a [pull request](https://github.com/albertomontesg/davis-interactive/pulls).

## Prediction Submission

After each iteration, it is mandatory to submit tha scribbles to evaluate and at the same time automatically generate the scribble for the next iteration.

```python
pred_masks = model.predict()
sess.submit_masks(pred_masks)
```

## Final Result

Once the session has finished a report can be asked using the `get_report` method. This method return a Pandas DataFrame where every row is the evaluation of every sequence, iteration and frame as well as the timing of every iteration. From this report, information of the performance against processing time can be extracted for comparison between interactive methods.
