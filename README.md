# ECS 171 2022 Fall Final Project

> **Note**
> 
> **Group 30**: 
> * Notebook used for previous submission: [Google Colab Notebook](https://colab.research.google.com/drive/1AqbJA3umP6qhSGoXuhuWKKh4dS5_917D?usp=sharing)
> * Notebook used for final submission: [Google Colab Notebook](https://colab.research.google.com/drive/1FxCVbPT-rJyXKdZTW4e3w5YCoAarldDA?usp=sharing)

<img src="https://thumbs.gfycat.com/GlumWastefulAdouri-max-1mb.gif">

The dataset is about car accidents in the USA from 2016 to 2021, which was collected using multiple Traffic APIs ([Reference](https://smoosavi.org/datasets/us_accidents)). 

## Abstract
Car is the most popular and convenient transportation nowadays, bringing accidents to people’s lives. Worldwide, approximately 1.3 million people die in car accidents each year. However, the accident could be avoided if we knew what factors would increase the possibility of a car accident. Therefore, in our project, we decided to analyze the US Accidents (2016 - 2021) dataset, by utilizing the dataset of car accidents covering 49 states of the USA, this project explores the factors that affect car accidents. These factors, such as weather, humidity, wind speed, weather conditions, etc, have a significant impact on severity. This project develops uses EDA to visualize data and help us better understand it, and finally build reliable models to predict it. Three supervised learning methods, including logistic regression, neural network, and random forest will be used to evaluate the relationships between variables.


## Introduction
Receiving the conveniences brought by cars, cars have become the most popular transportation tool today. Worldwide, approximately 1.3 million people die in car accidents each year. Since many people died because of car accidents, this problem cannot be neglected. However, the accident could be avoided if we knew what factor would increase the possibility of a car accident. Therefore, in our project, we decided to analyze the US Accidents (2016 - 2021) dataset. Factors such as weather, humidity, wind speed, and weather conditions will be taken into consideration in this project. Not only the natural factors, but also we are going to see whether human factors will be needed taken into account. The first thing we need to do is EDA, Exploratory Data Analysis. This part can let us see the difference between what we expect and actual data distribution. Before building our model, we will first do the data preprocessing to remove some unimportant factors. After data is processed, it is important that we build a good predictive model to investigate the underlying cause of car accidents in order to prevent any future car accidents.This project uses machine learning models, logistic regression, Random Forest Regressor,  and neural networks.


## Method

### Data Exploration
From the heatmap, we can see that there is no sepcific feature that has a strong correlation with `Serverity`.

<img width="1516" alt="Screenshot 2022-12-07 at 9 34 15 PM" src="https://user-images.githubusercontent.com/76938794/206364942-055c5bdf-dda4-48cd-8a79-80ec39f869c0.png">


From the two plots below, we see that Miami is the city that has the most number of car accidents and California is the state that has most accidents.

<img width="878" alt="Screenshot 2022-12-07 at 8 53 58 PM" src="https://user-images.githubusercontent.com/76938794/206364774-fae1250d-837d-443d-b550-5a7e88fffc1e.png">
<img width="861" alt="Screenshot 2022-12-07 at 8 54 10 PM" src="https://user-images.githubusercontent.com/76938794/206364777-cb840e7c-77a5-447d-9f6a-c0286f3f964f.png">


Majority of the car accidents collected in the dataset are from 2021 (53.1% of overall).
<img width="867" alt="Screenshot 2022-12-07 at 9 33 42 PM" src="https://user-images.githubusercontent.com/76938794/206364888-68c6ee5c-6bd7-4d3a-8e2d-892201034965.png">


Severity per year
<img width="987" alt="Screenshot 2022-12-07" src = "https://user-images.githubusercontent.com/118643840/206361875-32799656-9e00-418b-b624-04e000af300d.png">

### Data Preprocessing
We decide to delete some categorical variables because they won't affect the severity of car accidents. In addition, in order to futher explore the reason behind a car accident, we will add some new variables, such as the driving time and the terrain, which are consisting of the `Start_Time` ,`End_Time`, `Start_Lat`, `End_Lat`, `Start_lng` and `End_lng`. What's more, considering the percentage of missing value, we decide to delete any variables which its percentage is over 15%.

Colunms to drop
1.   `ID`
2.   `Description`
3.   `Airport_Code`
4.   `Timezone`
5.   `Start_Time`
6.   `End_Time`             
7.   `Start_Lat`              
8.   `Start_Lng`             
9.   `End_Lat`            
10.  `End_Lng`
11.  `Street`
12.  `Country`
13.  `Weather_Timestamp`
14.  `City`
15.  `Zipcode`

Based on Null and missing values percentage, we also will delete:
1. `Number`
2. `Precipitation(in)`
3. `Wind_Chill(F)`

Given that we have over 2 millions observations, we removed the missing values of the following labels: `Civil_Twilight`, `Nautical_Twilight`, `Astronomical_Twilight`, and `Sunrise_Sunset`.

#### Dropping the NaN rows(We have very large dataset)
Given that we have 2 millions observations and only a few thousands of rows have missing value, it is okay to drop them.

#### Feature Engineering -- Wind Direction
In `wind direction column`, we observe that there are some duplicative values, such as SSE and SSW, which they can be reclassified to south so that we can turn them into dummy variables in future.

#### Feature Engineering -- Weather Condition
In order to simplify the `weather_condition` column, we look up some online resources.
According to road weather management program, it tells that snow/sleet, rain and fog are the main weather condition cause car accidents. Therefore, we are only going to focus on these weather conditions.

#### Feature Engineering -- Duration

```
data['End_Time'] = pd.to_datetime(data['End_Time'])
data['Start_Time'] = pd.to_datetime(data['Start_Time'])
data['Duration'] = data.End_Time - data.Start_Time
new_data['Duration'] = new_data['Duration'].dt.total_seconds()
new_data['Duration'] = new_data['Duration'] / 60
```

#### Convert catorical variables using Orinal Encoder
For `Wind_Direction`, we categorize wind direction to main 4 directions: north, west, south and east
For `Weather_Condition`, the description about weather is really in detail. After some researches, we decided to categorize weather condition into 6 main categories: rain, fog, snow, cloud, clear, and thunderstorms. These are the main causes in car accidents.
For columns contain boolean values and strings, such as `state` and `side`, we converted them to dummy value using ordinal encoder.
```
ord_enc = OrdinalEncoder()

new_data.iloc[:,11:27] = ord_enc.fit_transform(new_data.iloc[:,11:27]).astype(int)
new_data.iloc[:,29:] = ord_enc.fit_transform(new_data.iloc[:,29:]).astype(int)
```

#### Dependent variable change
By observing the severity per year graph, we find that there is no `severity = 1` in any of the years except year 2020. Therefore, we decide to combine it with `severity = 2` and classify it as a level 1 traffic accident, which we also call a minor traffic accident. In addition, we also rename `severity = 3` and `severity = 4` to 2 and 3 respectively.

```
new_data.loc[new_data['Severity'] == 2, 'Severity'] = 1
new_data.loc[new_data['Severity'] == 3, 'Severity'] = 2
new_data.loc[new_data['Severity'] == 4, 'Severity'] = 3
```


#### Test and Split
Instead of spliting the data base on certain percentage, we decided to split data base on the year the accidents occur.
We use accidents happened before 2020 as training data, and accidents happened in 2020 as testing data. 
```
training_data = new_data[new_data["year"] < 2020].sample(frac = 0.5, random_state = 42)
test_data = new_data[new_data["year"] == 2020].sample(frac = 0.5, random_state = 42)
training_data = training_data.drop(['year'], axis = 1)
test_data = test_data.drop(['year'], axis = 1)
```

#### Models

##### Model1-Logistic Regression:
###### 1. We would use the Logistic Regression model from Sklearn. 
###### 2. First, we set up the model of logistic regression. We employ X_train and y_train to fit into the model. 
###### 3. Then, we extract the features and weight.  
###### 4. We also make a histogram showing the weight of each feature (feature weight). 
###### 5. Finally, we show the precision and recall through the classification report tool from Sklearn.

##### Model2-Random Forest Regressor:

###### 1. We use the Random Forest Regressor from Sklearn.
###### 2. First, we set up the model Random Forest Regressor. 
###### 3. We use the model to fit X_train and y_train. 
###### 4. Here, we extract the features and weight(feature weight). 
###### 5. We make a histogram showing the weight of each feature as well. 
###### 6. Finally, we use a classification report tool from Sklearn to show precision and recall.


##### PCA:

###### 1. First, we set up PCA through Sklearn and fit the X_train. 
###### 2. Then, we extract the labels and explained variance ratio from the PCA. 
###### 3. Then, we plot a graph that can show the explained variance in the PCA so that we can tell feature weight.

##### Model3-Artificial Neural Network:

###### 1. We use keras library to build up the artificial neural network. 
###### 2. For the first dense layer, based on input data, we set the input dimension to be 32, units to be 32, and activation function is ‘relu’. 
###### 3. The second and third dense layers have the same activation function ‘relu’ but different units, which are 16 and 8 respectively. 
###### 4. The last dense layer, as an output layer, has 4 units and takes the ‘softmax’ activation function. Here we use optimizer ‘adam’ and categorical_crossentropy as the loss. Also, we add metrics to measure accuracy. 
###### 5. Finally, we fit X_train and y_train into a model with batch size as 1000 and 10 epochs. In the below, we make a fitted graph to show how this model will fit.



## Result

### Logistic Regression Model
Since we try to find out what is the main factor of car accidents, we do a logistic regression model and make a graph on the weight of each feature.
```
model = LogisticRegression()
model.fit(X_train, y_train)
importances = pd.DataFrame(data={'Feature': X_train.columns,'Weight': np.abs(model.coef_[0])})
importances = importances.sort_values(by='Weight', ascending=False)
```

<img width="858" alt="Screenshot 2022-12-07 at 9 35 55 PM" src="https://user-images.githubusercontent.com/76938794/206365499-6fc08760-4604-4b73-97c5-13013744f1bf.png">
<img width="579" alt="Screenshot 2022-12-07 at 9 36 01 PM" src="https://user-images.githubusercontent.com/76938794/206365507-957e3b90-118d-4aca-9b14-ca9e05b7e18b.png">

### Random Forest Regressor 
```
importances = pd.DataFrame(data={'Feature': X_train.columns,'Weight': np.abs(model.feature_importances_)})
importances = importances.sort_values(by='Weight', ascending=False)
```
<img width="872" alt="Screenshot 2022-12-07 at 11 40 53 PM" src="https://user-images.githubusercontent.com/76938794/206389362-fbb506eb-4456-4bf8-b622-64f26f2950e8.png">

### PCA
```
labels = [p for p in range(1,len(pca.explained_variance_ratio_.cumsum())+1)]
importances = pd.DataFrame(data={'PCA': labels,'explained_variance': pca.explained_variance_ratio_.cumsum()})
```

<img width="894" alt="Screenshot 2022-12-07 at 11 40 45 PM" src="https://user-images.githubusercontent.com/76938794/206389520-5367a869-5872-4017-be46-8de37f11f696.png">

<img width="1363" alt="Screenshot 2022-12-07 at 11 40 35 PM" src="https://user-images.githubusercontent.com/76938794/206389527-4590ed65-be8d-447b-8f29-9ffe52f0fd94.png">


### Neural network Model
For the first three layers, we used 32, 16, and 8 units respectively; and activation function 'relu'.
For the output layers, we used 4 units and 'softmax' as the activation.
```
nn_model = Sequential()
nn_model.add(Dense(units = 32, activation = 'relu', input_dim = 32))
nn_model.add(Dense(units = 16, activation = 'relu'))
nn_model.add(Dense(units = 8, activation = 'relu'))
nn_model.add(Dense(units = 4, activation = 'softmax'))
nn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
```

<img width="496" alt="Screenshot 2022-12-07" src="https://user-images.githubusercontent.com/118643840/206365929-8d505cf6-e33d-4cab-a16d-49fbcb030aa3.png">
<img width="495" alt="Screenshot 2022-12-07" src="https://user-images.githubusercontent.com/118643840/206366099-192dfe0f-dd8e-4ce2-a168-beb1f54051cf.png">
<img width="567" alt="Screenshot 2022-12-07" src="https://user-images.githubusercontent.com/118643840/206366205-816d2d52-4210-4ba1-a856-ed3a86a0c46c.png">

For our neural network model, we were able to achieve accuracy of 83%.

<img width="496" alt="Screenshot 2022-12-07" src = "https://user-images.githubusercontent.com/118643840/206369477-bd979aa7-75a7-43b8-b11e-ae632d385ade.png">


### How is our model in fitting graph
Confusion Matrix
, training error vs test error. From the graph, it looks not good because the training and testing line are far away from each other. It shows a overfitting.

## Discussion
Initial data exploration can let us see the difference between speculation and real data. From our interpretation, we think car accidents happen more on rainy days than on sunny days. However, the result is not what we expected. The fair weather has the most car accidents. We think people are more aware of their safety on rainy days. On sunny days, people will drive faster to pursue a kind of alacrity.  Also, people will do other things while driving. Like on sunny days, if someone hears that a boss is calling him, he will definitely pick up his phone, which may lead to car accidents. 

The humidity result is what we expected. As we know, the more humidity the air is the less friction coefficient between the road and the wheel. Since friction coefficient becomes lower, the friction force decreases, which leads to the car harder to be broken down.  This will lead to more car accidents. 
That is the importance of data analysis. As long as we use data to show the degree of each factor contributing to the car accident severity, we can really conclude what and how each factor affects car accidents. We cannot take anything for granted that rainy days will affect car accidents or any other speculation. 

For our modeling, we choose neural network, logistic regression, and random forest because we have a very large dataset. Compared with the other traditional machine learning methods, a neural network requires much more data because it needs to be fed by the data to update the weights,  thus finding the best weight. However, the memory is one of the challenges while we do the training because we need to store the weights and do the backpropagation. In the model training, we only use a small portion of the dataset. From the classification reports of our logistic regression model and neural network, both models did an excellent job of predicting car accidents when severity is 2. In logistic regression, we got an 88% accuracy and an 83% accuracy for the neural network model. Compared with the other models we made, we find that neural network is sensitive to the severity of car accidents with level 2 and 3. It makes some correct predictions. In addition, through logistic regression, random forest and principal component analysis, it is found that distance, wind speed, pressure, duration and some weather conditions are the main factors causing traffic accidents.

After the data preprocessing and modeling, we find that the dataset is pretty unbalanced, since over 90% of the severity of car accidents is level 1. It also indicates that the minor car accidents take a large portion of the dataset. It is also a good sign that there is not too much insane driving in the states. However, this results in a lack of data to make a reliable model to predict what conditions will cause a very serious car accident. Due to the given data, we suspect that the severity of car accidents with level 2 and 3 may be caused by some unnatural factors. We can also conclude that most people will drive carefully when the natural condition is not good.

In the future data collecting, we suggest adding some human factors, such as driving fatigue, distraction, driving experience, drug history and so on. Also, the car history data is also important, such as repair history, damages, and so on. Or, we can collect car accident images and focus on the visuals of car accidents so that we can use data preprocessing and convolutional neural networks to fit these visuals.



## Reference:
- https://stackoverflow.com/questions/38152356/matplotlib-dollar-sign-with-thousands-comma-tick-labels
- https://www.kaggle.com/code/nikitagrec/usa-accidents-plotly-maps-text-classification
- https://www.kaggle.com/code/tusharsingh1411/us-road-accident-eda
- https://www.atmosera.com/blog/binary-classification-with-neural-networks/
- https://betterdatascience.com/feature-importance-python/

## Contribution
### Author: 
- Yinyin Guan: worked on EDA, graphs, intro 
- Randy Li: modify abstract, intro, and write some code of models, but fails.   
- Shuying Li: worked on EDA, Random Forest, Logistic Regression, PCA  
- Zuge Li: delete useless variables like turning loop. Help data preprocessing by converting time scale.  build up artificial neural network, plot fitted graph, writing method model part  
- Hugo Lin: worked on the writing portion of data preprocessing in readme, make plot for testing and training accuracy  
- Jianfeng Lin: data preprocessing, neural network modeling, discussion(model finding , future suggestion)  

---
