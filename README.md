# fakenews
fake news detection
Fake News Detection
Fake News Detection in Python
In this project, a variety of natural language processing techniques and machine learning algorithms have been employed to classify fake news articles utilizing sci-kit libraries from Python.

Getting Started These instructions will guide you to obtain a copy of the project and run it on your local machine for development and testing purposes. Refer to the deployment section for notes on how to deploy the project on a live system.

Prerequisites The following items need to be installed and instructions on how to do so are provided:

Python 3.6 Installation of Python 3.6 is necessary for this setup. Please refer to this URL https://www.python.org/downloads/ to download Python. Once downloaded and installed, PATH variables must be set up to run Python programs directly. Detailed instructions can be found at https://www.pythoncentral.io/add-python-to-path-python-is-not-recognized-as-an-internal-or-external-command/. Setting up PATH variables is optional as the program can run without it. Further instructions on this topic are provided below. Alternatively, Anaconda can be downloaded, and its anaconda prompt can be used to run the commands. To install Anaconda, refer to this URL https://www.anaconda.com/download/.

After installing either Python or Anaconda, the following three packages must be downloaded and installed:

Sklearn (scikit-learn) numpy scipy

If Python 3.6 was installed, run the following commands in the command prompt/terminal to install these packages: pip install -U scikit-learn pip install numpy pip install scipy

If Anaconda was installed, run the following commands in the Anaconda prompt to install these packages: conda install -c scikit-learn conda install -c anaconda numpy conda install -c anaconda scipy

Dataset Used The LIAR dataset was used for this project, which contains three files with tsv format for test, train, and validation. Below is some description about the data files used for this project.

LIAR: A BENCHMARK DATASET FOR FAKE NEWS DETECTION

William Yang Wang, "Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection, to appear in Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL 2017), short paper, Vancouver, BC, Canada, July 30-August 4, ACL.

The original dataset contained 13 variables/columns for train, test and validation sets, as follows:

Column 1: the ID of the statement ([ID].json). Column 2: the label. (Label class contains: True, Mostly-true, Half-true, Barely-true, FALSE, Pants-fire) Column 3: the statement. Column 4: the subject(s). Column 5: the speaker. Column 6: the speaker's job title. Column 7: the state info. Column 8: the party affiliation. Column 9-13: the total credit history count, including the current statement. 9: barely true counts. 10: false counts. 11: half true counts. 12: mostly true counts. 13: pants on fire counts.

Column 14 pertains to the context, specifically the venue or location of the speech or statement. To simplify matters, only two variables have been selected from the original dataset for the purpose of classification. Additional variables may be incorporated at a later time to increase complexity and augment features.

The following columns were utilized to generate three datasets employed in this project:

Column 1: Statement (News headline or text). Column 2: Label (Label class contains: True, False)

It is noted that the newly created dataset only contains two classes, in contrast to the original six classes. The method utilized to reduce the number of classes is as follows:

Original -- New True -- True Mostly-true -- True Half-true -- True Barely-true -- False False -- False Pants-fire -- False

The datasets used in this project were in CSV format, specifically train.csv, test.csv, and valid.csv, which can be found in the repository. The original datasets are in TSV format, located within the "liar" folder.

File descriptions are as follows:

DataPrep.py: This file contains all the preprocessing functions necessary to process input documents and texts. Initially, the train, test, and validation data files are read, followed by pre-processing steps such as tokenizing and stemming. Furthermore, exploratory data analysis is conducted, including response variable distribution and data quality checks such as null or missing values.

FeatureSelection.py: In this file, feature extraction and selection methods from the Sci-Kit Learn Python libraries are employed. For feature selection, methods such as simple bag-of-words and n-grams are utilized, followed by term frequency such as TF-IDF weighting. Word2vec and POS tagging are also utilized to extract features, although they have not been utilized at this stage in the project.

classifier.py: In this file, all classifiers for predicting fake news detection are built. The extracted features are fed into different classifiers, including Naive-Bayes, Logistic Regression, Linear SVM, Stochastic Gradient Descent, and Random Forest Classifiers from Sci-Kit Learn. Each of the extracted features is used in all of the classifiers. After fitting the model, the F1 score is compared, and the confusion matrix is checked. Following the fitting of all classifiers, two best-performing models are selected as candidate models for fake news classification. Parameter tuning is conducted through implementation of GridSearchCV methods on these candidate models, and the best-performing parameters for these classifiers are chosen. The selected model is then utilized for fake news detection, with the probability of truth being determined. In addition, the top 50 features are extracted from the term-frequency TF-IDF vectorizer to determine the most important words in each of the classes. Precision-Recall and learning curves are also employed to determine how the training and test set performs as the amount of data in our classifiers increases.

Our finalized and top-performing classifier was the Logistic Regression model, which was then saved on disk under the name final_model.sav. Once this repository is closed, the model will be transferred to the user's machine and utilized by the prediction.py file to classify fake news. The program takes a news article as input from the user, and then the model is used for the final classification output, which is then displayed to the user, along with the probability of truth.

The project's process flow is shown below:

Performance: The learning curves for our candidate models are shown below:

Logistic Regression Classifier

Random Forest Classifier

Next Steps: As we observed that our best-performing models had an f1 score in the 70's range, this is due to the limited amount of data we used for training purposes and the simplicity of our models. For future implementations, we could introduce additional feature selection methods such as POS tagging, word2vec, and topic modeling, as well as increasing the training data size. We plan to implement these techniques in the future to improve the accuracy and performance of our models.

Installation and Steps to Run the Software: To get started, the first step would be to clone this repository into a folder on your local machine. To do so, you need to run the following command in the command prompt or git bash:

$ git clone https://github.com/nishitpatel01/Fake_News_Detection.git

This will copy all the data source files, program files, and models onto your machine. If you have chosen to install anaconda, follow the instructions below. After saving all the files to a folder on your machine, if you have chosen to install anaconda from the prerequisites section, open the anaconda prompt, change the directory to the folder where this project is saved on your machine, and type the following command and press enter:

cd C:/your cloned project folder path goes here/

Once you are inside the directory, call the prediction.py file. To do this, run the following command in the anaconda prompt:

python prediction.py

After hitting enter, the program will prompt you for an input, which will be a piece of information or a news headline that you want to verify. Once you have entered or pasted the news headline, press enter.

Once you hit enter, the program will take the user input (news headline) and use the model to classify it into one of the categories of "True" and "False". In addition to classifying the news headline, the model will also provide a probability of truth associated with it.

If you have chosen to install python (and did not set up the PATH variable for it), follow the instructions below:
cd C:/your cloned project folder path goes here/
After cloning the project to a folder on your machine, it is necessary to open the command prompt and navigate to the project directory by executing the following command: "cd C:/your cloned project folder path goes here/". To locate the python.exe file, you can use the search bar in the Windows Explorer. After finding the path to the python.exe file, it is important to write the entire path of the file followed by the entire path of the project folder with prediction.py at the end. For example, if the python.exe file is located at c:/Python36/python.exe and the project folder is located at c:/users/user_name/desktop/fake_news_detection/, then the command to run the program will be as follows: "c:/Python36/python.exe C:/users/user_name/desktop/fake_news_detection/prediction.py". Once the command is executed, the program will prompt the user for input, which will be a piece of information or a news headline that needs to be verified. After entering the news headline, the user can press enter.

After pressing enter, the program will use the user input (news headline) and the model to classify the statement as either "True" or "False". Additionally, the model will provide a probability of truth associated with the statement. It may take a few seconds for the model to classify the given statement, so the user should be patient.

If python has been installed (and the PATH variable for python.exe has been set up), follow these instructions: open the command prompt and navigate to the project folder by executing the following command: "cd C:/your cloned project folder path goes here/". Then, run the following command: "python.exe C:/your cloned project folder path goes here/". After executing the command, the program will prompt the user for input, which will be a piece of information or a news headline that needs to be verified. After entering the news headline, the user can press enter.

After pressing enter, the program will use the user input (news headline) and the model to classify the statement as either "True" or "False". Additionally, the model will provide a probability of truth associated with the statement.
