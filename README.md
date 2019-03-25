# Post-Editing-Prediction-on-Stack-Overflow
Related paper: 
https://drive.google.com/file/d/1mv5TgAioB4DbdCWyDMrNaZi9_3WNaYFr/view?usp=sharing

The project basically includes an empirical study, data preprocessing, training and testing process
for CNN and other baseline methods (Logistic Regression, SVM, Fasttext and attention-based_LSTM), as well
as an visualization approach that mentioned in the report.

Software and Hardware Configurations： 

Windows 10  
Nvidia P40 GPU (24g Memory)  
Python 3.6.4  
tensorflow-gpu 1.11.0  
Keras 2.2.4  
numpy 1.14.5  

To implement the code of CNN and models in the control group, you need first to download data from the following links:

   https://drive.google.com/open?id=18bXlswd6zexIW66Yzc_XSysFuKNJPINv  
   https://drive.google.com/open?id=1Cd9umhzIkTGk9W6DHyC6drhXBunbWWB9  
   https://drive.google.com/open?id=1Ej6B8250DOYu-YCzExI3-EWvz88JE936  
   https://drive.google.com/open?id=1pCD9xhtfCaoltTT8U_HA--Rwj-LeUW6D  

Download these files (data.rar, vocabulary.rar, splitted_data.rar and models.rar) and extracted them to the artefacts directory.

The command to train and test CNN model:

    python main.py --pattern content --batchsize 64 --epoch 10

where pattern is the edit type , including content, image, link, format, which respectively represents text format, 
image, link and code format.

The command to create the html for visualizing the prediction result of a post example:

    python visualisation.py

An html file with filename pattern + 'visualization.html' will be generated under the local directory.

To implement baseline methods, run the train.py file under control_group_model directory.  
Note that this file can only run after CNN is build as the baseline methods have to reuse the dataset that CNN generated.

train.py will implement attention-based LSTM, SVM and Linear Regression one by one for a specific edit type.
The fasttext has been commented out as it is not supported by windows and the method can only run on a removed
Mac computer.

Empirical_study directory contains the code of computing the n-gram frequency (unigram to trigram) and LDA modelling
method. The PostHistory.xml file is a very large dataset with approximately 99G. It is hard to upload the data we used
onto geogle drive or dropbox due to the data limit. However, there is the link for downloading the updated post history:
(Our dataset containing all post history before Aug. 2017 while the data provided here have been updated to Sept. 2018)

   https://archive.org/download/stackexchange/stackoverflow.com-PostHistory.7z

You can uncompress the .7z file into Empirical_study directory and rename it as Post History.xml. Then run ngram_frequency.py 
to gain the top 100 frequency n-grams or run LDA_modeling to see the 8-topic modeling results summarized from post history 
data.
