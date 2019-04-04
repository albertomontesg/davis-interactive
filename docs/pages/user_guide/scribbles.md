# Scribbles Default Format

The scribbles are stored in a key-value dictionary which can be encoded in a `json` file or a dictionary object. The body of this dictionary should have the following fields:

```json
{
    "scribbles": [...],
    "sequence": "sequence-name",
    "annotated_frame": 10
}
```

The fields represent the following:

* `scribbles`: a list of length equal to the number of frames of the sequence. For each frame there is a list of all paths or lines of the scribbles
* `sequence`: sequence name of the scribble.
* `annotated_frame` (optional): number of the frame that is annotated for fast lookup.

The lines of the scribbles should be stored as follows:

```json
{
    "scribbles": [
        [],
        [],
        ...
        [],
        [{
           "path": [[x, y] * nb_points],
           "object_id": 0,
           "start_time": 0,
           "end_time": 1000, // 1000ms = 1s
        }, {
           "path": [[x, y] * nb_points],
           "object_id": 1,
           "start_time": 2000,
           "end_time": 3000,
        }], // Annotated frame
        [],
        ...
        [],
        []
    ],
    "sequence": "sequence-name",
    "annotated_frame": 10
}
```

