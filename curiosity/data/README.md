# New dataset!

This writeup is for the tfrecords written off of the hdf5 containing original data. tfrecords are urrently located in `/mnt/fs0/datasets/two_world_dataset`, with folders new_tfdata and new_tfvaldata for training and validation.

# Image-like attributes

Attributes `images`, `normals`, `objects`, `images2`, `normals2`, `objects2` contain attributes that are shape (160, 375, 3) with datatype `uint8`. `images` contain images from the main camera view, `normals` the corresponding normals, `objects` a 256-base id encoding (channel 2 is 256^0...) of pixels at which the object with the assigned id appears (ids are unique for a scene, and hence within a batch, but not across different batches). `images2`, `normals2`, `objects2` are the same but for camera position 2 (above and looking down).

# Actions

The attribute `actions` is of shape (9,) and datatype float32, and `actions2` is of shape (7,) and datatype float32. It is a concatenation of (force rotated to main camera coordinates, torque rotated to main camera coordinates, position, id of object force applied to). Position is simply guaranteed to be a pixel location at which the object acted on is. 'actions2' corresponds to camera position 2, and it excludes the position of the action (which can still be found via `object_data2`). Id is -1 and everything else is zero if no action is being applied.

# Object_datas

The attributes `object_data` and `object_data2` are of shape (11,12) and datatype float32. Again organized by camera, each contains (id of object, pose quaternion relative to camera, 3d position relative to camera, 3d position projected onto screen -- 2 dimensional, centroid of pixels corresponding to object). Objects are ordered as follows: object acted on (constant between two teleports -- that is, even if no action is being applied, it is the object that had or will have action applied in the current between-teleport sequence), 10 objects with the most pixels in the main camera view. If object data is not available, zeros are entered (this is coded in indicators below)

NB: This is mostly a convenient way of organizing much more data from the original dataset write and is relatively easy to change if something differently explicit is more useful.

# Agent_data

`agent_data` is shape (6,) and type float32. It is (absolute main camera position, absolute main camera pose in Euler coordinates).

# Reference_ids

`reference_ids` is size 2 and type int32. It is a unique identifier of each frame, specifically (hdf5 file index -- defined in `make_new_tfrecord.py` -- it came from, frame number of hdf5 file)

# Indicators

All are shape (1,) and type int32. The dataset consists of the agent performing a sequence of teleport, drop object (sometimes), apply action, wait.

`is_acting` is 1 if an action is currently being applied, 0 else.
`is_not_dropping` is 1 if the object is dropping (i.e. every frame between the teleport and when an action is applied), 0 else.
`is_not_teleporting` is 1 if the agent/objects are not being teleported.
`is_not_waiting` is 1 if the agent is not waiting after an action.
`is_object_in_view` is 1 if the object appears in the main camera.
`is_object_in_view2` is 1 if the object appears in camera 2.
`is_object_there` is 1 if we have position and pose information of the object -- if the object gets too far away after an action is applied, for instance, we lose this information.

# File naming convention

Examples: TABLE:NODROP:LONG_PUSH:0:275.tfrecords and ONE_OBJ:FAST_PUSH:3:193.tfrecords

If placing an object on another object: staging_type:drop or no drop done:action_type:number of batch in a scene:an additional index to make this unique

If just with one object: staging_type:action_type:number of batch in a scene:additional index.

staging_type is how objects are staged during a teleport. E.g. ONE_OBJ is just the agent teleporting to one object, to which it applies an action, TABLE is an object placed on a table, TABLE_CONTROLLED is an object placed near the edge of the table, with the agent z-view roughly parallel to the edge (actions are applied roughly in the direction of the edge). ROLLY_ON_TABLE is like TABLE but where synsets that have a higher likelihood of being round are used.

action_type is a description of the action type applied in these batches. Magnitudes and directions are random in ways that vary by action_type and staging_type. E.g. LONG_PUSH is a horizontal push applied for several (5-10) frames, and FAST_PUSH is a (generally stronger) push applied for 1 frame.

number of batch in a scene: The data is organized into scenes, which represent spawning of new objects in new initial positions. The staging_type is constant in a scene, but roughly, stuff can get more cluttered around tables as the scene goes on, so higher numbers might correspond to a higher complexity of interaction.

