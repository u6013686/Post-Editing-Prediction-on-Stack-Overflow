# Post-Editing-Prediction-on-Stack-Overflow

The project basically includes an empirical study, data preprocessing, training and testing process
for CNN and other baseline methods (Logistic Regression, SVM, Fasttext and attention-based_LSTM), as well
as an visualization approach that mentioned in the report.

## Related paper

https://chunyang-chen.github.io/publication/proactiveEdit_cscw18.pdf

## Configuration 

Windows 10  
Nvidia P40 GPU (24g Memory)  
Python 3.6.4  
tensorflow-gpu 1.11.0  
Keras 2.2.4  
numpy 1.14.5  

## Data

To implement the code of CNN and models in the control group, you need first download data from the following links:

   https://drive.google.com/open?id=18bXlswd6zexIW66Yzc_XSysFuKNJPINv  
   https://drive.google.com/open?id=1Cd9umhzIkTGk9W6DHyC6drhXBunbWWB9  
   https://drive.google.com/open?id=1Ej6B8250DOYu-YCzExI3-EWvz88JE936  
   https://drive.google.com/open?id=1pCD9xhtfCaoltTT8U_HA--Rwj-LeUW6D  

## Implementation

The command to train and test CNN model:

    python main.py --pattern content --batchsize 64 --epoch 10

where pattern is the edit type, including content, image, link, format, which respectively represents text format, 
image, link and code format.

Examples of typical edits are shown below:

![image](https://github.com/u6013686/Post-Editing-Prediction-on-Stack-Overflow/blob/master/edittype.png)

The input of each binary classification model is posts before edited and the target output is whether the post should be edited or not.

The command to create the html for visualizing the prediction result of a post example:

    python visualisation.py

An *html* file with filename *pattern + visualization.html* will be generated under the local directory, in which there is an example of attention visualization.

sample visulization result:

![image](https://github.com/u6013686/Post-Editing-Prediction-on-Stack-Overflow/blob/master/visualization.png)

To implement baseline methods, run *train.py* file under *control_group_model* directory.  
Note that this file can only run after CNN is build as the baseline methods have to reuse the dataset that CNN generated.

*train.py* will implement attention-based LSTM, SVM and Linear Regression one by one for a specific edit type.
The fasttext has been commented out as it is not supported by windows and the method can only run on a removed
Mac computer.

*Empirical_study* directory contains processes of computing the n-gram frequency (unigram to trigram) and LDA topic modelling
method. Data source is available in:

   https://archive.org/download/stackexchange/stackoverflow.com-PostHistory.7z
   


