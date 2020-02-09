# Project - Ally

### Introduction:
- Machine Learning is one of the most influential and powerful technology in today's world. Most importantly still its state-of-art concept we are far from its full potential. When it comes to developing machine learning algorithms for specific business problem most of us will have below questions running in your mind. 
	- Which algorithm to begin with for my business problem.
	- Which is the best suitable algorithm for my business problem.
	- What is the Optimized tuning parameter for my business problem.
	- Which set of input variables gives me more accurate result and so on.

- As a Data Scientist it is our responsibility to pick a best algorithm and deliver high accurate models in short period of time. If the use case is simple like predicting the Binary classification [Spam or Not Spam] we can go ahead and develop a solution for the business problem. but if the data is too dirty and your finding difficult to build your model with accurate result don't worry we are here you have **Ally**. 

- **Ally** is Intended to solve all kind of business problem in Data Science/Machine Learning era. yes it can help you to build your model for machine learning problems like Regression, Classification, Clustering and etc. all thing you need to do is simply feed your data-set with minimal configurable parameter and run "PA" will build the best suitable model for your data-set with Optimized tuning parameter configuration and it will give you the model file as output. now you can use this generated model in your client program and deploy it in any environment. Its that simple.


### How to Run:
- Running Ally is pretty simple. you have to use RunModel() method from run_model.py to trigger the job. which takes few parameters such as job_type, input_path and target variable.   
```py 
# Object Initialization
obj = RunModel(job_type='Regression', input_path=file_path, target='MPG')
```   

- Most of the parameters were defaulted please find the below list of parameters you can play with it.
	- job_type: Type of model that you are trying to apply. It could be any one of below.
		- Regression
		- Classification
	- input_path: path to load input file to be loaded
	- target: Target variable [Dependent Variable(Y)]
	- features: Features of matrix of X [Independent Variable]
	- algorithms: list of algorithms to be applied according to job_type, it can be anyone from below
		- ALL --> ALL IN
		- MLR --> MultiLinearRegression
		- POLY --> PolynomialRegression
		- SVR --> SupportVectorRegression
		- DTREE --> DecisionTreeRegression & DecisionTreeClassification
		- RFR --> RandomForestRegression
		- SIGMOID --> LogisticRegression
		- KNN --> K-NearestNeighbours Classifier
		- SVM --> SupportVectorMachine Classifier
		- RFC --> RandomForestClassifier
		- BAYESIAN --> NaiveBayesClassifier
	- file_format: input file format to be read
	- missing_value: True, if data-set contains any missing value, default to False
	- search_value: applicable when missing_value is True, default to None
	- feature_selection: Apply Feature selection on Regression model which follows the below feature selection procedures. for feature selection we are using statsmodel library, it is a iterative approach, by default it is None
		- BE: BackWard-Elimination
		- FS: Forward-Selection
	- feature_scaling: scale matrix of feature of X, default to False, it must be true when feature_selection is enabled
	- encode_categorical: encodes categorical data to integer format when it sets True, default to False
	- binary_transform: binarize encoded categorical data to binary format, default to False
	- categorical_features: list of categorical features to be encoded
	- binary_classifier: defines whether the target variable is Binary or not, defaulted to False.
		- If true Logistic Regression would take into consideration.
		- If false remaining classifiers only considered


- Once you initialized the model it very simple to run it by simple running the below command.
```py
obj.run()
```
- to get model results please run the below command. if you want result of any specific model you can pass the model name as input mentioned below.
```py
obj.get_summary()
or
obj.get_summary('MLR')
```