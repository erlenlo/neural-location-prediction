### Introduction

This code is used to collect twitter datasets using the Twitter streaming API, preprocess the content and train neural classifiers in order to predict the geographical location of tweets using the tweet text and other metadata.
The code is developed during a master's thesis at the Norwegian University of Science and Technology, Faculty of Information Technology and Electrical Engineering.

### Datasets

To generate datasets for training and testing run `grid.py` with appropriate input file. Example input file is shown below


| Latitude  | Longitude | Tweet text |
| :-------------: | :-------------: | :-------------: |
| 51.50821756  | 0.02840008  | Lorem ipsum dolor sit amet  |
| 51.50464224  | -0.01682281  | Duis aute irure dolor in reprehenderit in voluptate  |
| 51.50821756  | 0.02840008  | Excepteur sint occaecat cupidatat non proident  |

The output from `grid.py` will result in a grid file containing grid cells representing the geographical area, and a training and test set where each tweet is assigned to the correct grid cell.

Example grid file

| GCID  | SW latitude | SW longitude | NE latitude | NE longitude |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| 0	| 51.58280327	| 0.1279558	| 51.65352506	| 0.2005964 |
| 1	| 51.58139 | 0.18631 | 51.58139 | 0.18631 |
| 2	| 51.5793743 | 0.18550747 | 51.57939523 | 0.18566984 |

Example training / test file

| GCID | Latitude  | Longitude | Tweet text |
| :-------------: | :-------------: | :-------------: | :-------------: |
| 0 | 51.50821756  | 0.02840008  | Lorem ipsum dolor sit amet  |
| 1 | 51.50464224  | -0.01682281  | Duis aute irure dolor in reprehenderit in voluptate  |
| 2 | 51.50821756  | 0.02840008  | Excepteur sint occaecat cupidatat non proident  |

### Initialize the training and predict locations

* Step 1: Run `neural/run.py` with generated training set as dataset and desired settings for the neural network.
* Setp 2: Run `neural/prediction.py` with generated training and test set as datasets and generated model from previous step.
