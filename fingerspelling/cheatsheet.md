# File Processing

```Python
import os

os.path.join(path, "data", "raw")
```

# Github


Before editing files:

```sh
git checkout -b <branch-name, eg feature-encoding>
```

pushing

```sh
git push origin <branch-name, eg feature-encoding>
```

# Preprocessing
For preprocessing images for prediction, we need to pad the coordinates based on
min, max of x and y values from mediapipe

Then we need to normalize the coordinates based on total sqaure side length in
pixels


# pickle files
remember to label encode, not onehotencode
remember to under, over sample or smote
remember to train test split

# How to use the scripts
## fingerspelling.code_logic
### data.py
#### load_raw_data(IMAGES_PATH)
Takes an image path (folder).
This folder must include **Labelled folders** inside. (A, B, C, 0, 1, 2, etc)
Returns image_list, image_path_list, label_list

#### load_processed_data()
Set env variable PROC_DTYPE to either [black_image, landmark_coordinates]
black_image : unwritten

landmark_coordinates :
Set env variable LOAD_DATA_COOR to the pickle file
Pickle file will be read
Returns processed_data

### model.py
Only simple RandomForestCLassifier at the moment.

### preprocessor.py
#### draw_landmarks_on_image(rgb_image, detection_result)
Code provided by mediapipe that is used in get_black_image()
#### detection(image_path)
Input image path, returns detection_result object
Output is used in get_black_image()
#### get_black_image(detection_result, image_list)
Input detection_result from detection(), image_list from load_raw_data in data.py
Returns annotated_image
#### black_save_single(black_image, label_list, image_list, idx)
Writes single black_image into local directory
#### black_save(black_image, label_list, image_list, idx)
Writes black_images into local directory
#### get_landmark_coordinates(image_path)
Takes one image path and
Returns a list of landmark coordinates
Output to be converted into a pd.Dataframe for prediction
#### landmark_save(IMAGES_PATH, landmark_file_name)
Takes an image path (folder). landmark_file_name is a string, e.g. 'test'.
This folder must include **Labelled folders** inside. (A, B, C, 0, 1, 2, etc)
Saves data in a pickle file on local directory data/processed/coordinates.
Data can be loaded by using load_processed_data() in data.py

### registry.py
#### save_model(model)
Saves model locally.
#### load_model()
Loads latest model.

## interface
### main.py
