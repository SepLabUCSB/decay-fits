
# Decay-fits

A Python program designed to analyze nanoimpact electrochemistry spike-decay .txt files.

It returns the complete trapezoidal integral for each impact and the trapezoidal integral from the start of the spike
to the minimum.

Option to fit the decay with different functions and export scipy.optimize.curve_fit outputs.


## Installation

Clone the repo with GitHub.

```
  git clone https://github.com/SepLabUCSB/decay-fits

```
    
## Usage/Examples

Open decay-fits.py in your favorite Python environment (Example: Spyder).

In order to parse through multiple files and remove points. Set Matplotlib graphics Backend to Qt or type in the console:
```
%matplotlib
```


Set the variable data_folder to the folder path that includes the file to analyze. An example file is included in the repository with the name "Sample_spike.txt".
![App Screenshot](https://github.com/SepLabUCSB/decay-fits/blob/trunk/docs/Code_inputs.png?raw=true)

Run the program in the environment.

An interactive window opens, allowing you to zoom in on the spike, inspect it, and remove it with a single click.
![App Screenshot](https://github.com/SepLabUCSB/decay-fits/blob/trunk/docs/Remove_spike.gif?raw=true)

The buttons at the bottom of the window allow users to analyze the next file, go back, reset the current file if a point was removed mistakenly, and save the data as an .xlsx file. Each file can be exported individually, and a combined output will also be generated.


