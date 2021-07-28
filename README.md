# weather4cast2021_Stage1
3rd place solution for the Weather4cast 2021 Stage 1 Challenge

### Dependencies
The code can be executed from a fresh environment using the provided list of requirements: `conda env create -f environment.yml`.

### Inference
A script has been create to made predictions using a trained model on new data as per requirements detailed in competition: (#https://github.com/iarai/weather4cast#code-and-abstract-submission)

The model weights of the final submission for both core and transfer learning can be downloaded from https://drive.google.com/file/d/1mX4HMbf4QAW12DgAq1Bge1WT_5nAnz7N/view?usp=sharing

To run predictions on a test dataset use ('inference.py'). This should fine on a CPU machine

examples of usage:

    - inference for Region R1

    R=R1
    INPUT_PATH=data
    WEIGHTS=weights/Lomb_14.pth
    OUT_PATH=.
    python weather4cast/inference.py -d $INPUT_PATH -r $R -w $WEIGHTS -o $OUT_PATH -g 'cuda'



### Train/evaluate a UNet
To replicate the training for the 3rd place solutions we recommend use of the training notebook (`Training.ipynb`).

Alternatively, a single script has also been provided (`train.py`).

In the actual competitions, the authors actually included an extra step to preprocess all the data and save them all on disk. This reduced the training time.

