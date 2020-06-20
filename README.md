# Model Selection

## Introduction
In the following sessions, we will learn concepts and principles which are central to all of machine learning. In this session, we will build a conceptual foundation which will likely be useful in almost every machine learning problem we will solve in the future. 

After this module, we should be able to apply some fundamental principles to choose appropriate models and critically evaluate the pros and cons of each model. The topics and some important jargons in this session include:

* Occam’s Razor 
* Overfitting
* Regularization
* Bias-Variance Tradeoff
* Model Complexity

## Principles of Model Selection
In the previous course, you may have come across situations where a model performs well on training data but not on the test data. Also, you would have faced confusion about which model to use to a given problem. For example, by now you have learned many classification models. Given a problem that requires classification, how would you decide about the best one to go with?

Questions like these frequently arise irrespective of the choice of model, data or the problem itself. The aim of this session is to answer questions like these.

In these sessions you will be learning some thumb rules and some general pointers about how to go about selecting the appropriate models. This session is just a discussion on such thumb rules. In the subsequent sessions, you will see these rules being applied in the context of various problems and algorithms. The central issue in all of the machine learning is “how do we extrapolate learnings from a finite amount of available data to all possible inputs ‘of the same kind’?” Training data is always finite, yet the model is supposed to learn everything about the task at hand from it and perform well on unseen data.

Take the car pricing dataset in linear regression, for example, which was trained using a few thousand observations. How do you ensure, and be confident, that the model is as good as it seems on the training data and deploy it to make predictions on real, unseen data?

Often, it is mistaken that if a model performs well on the training data, it will produce good results on test data as well. Very often, that is not the case.

![title](image/model.png)

![title](image/model1.jpg)

**Occam's razor** is perhaps the most important thumb rule in machine learning, and incredibly 'simple' at the same time. **When in dilemma, choose the simpler model.** The question then is 'how do we define simplicity?'. In the next segment, you will study some objective ways to measure model simplicity and understand why simplicity is preferred over sophistication and complexity using various examples.

## Model and Learning Algorithm
Before you dive into what exactly a simple model is, what all its benefits are, we will take a short detour to reiterate some terminologies and the machine learning framework. You will now understand the process of using training data, learning from it and then building a model to describe a system which performs a task at hand, like classification or regression. The key objectives here are to understand:
* The meaning of model, learning algorithm, system and hypothesis class
* The (often misunderstood) difference between a learning algorithm and a model
* The meaning of ‘class of models’ 

![title](image/model2.JPG)

Learning algorithm's task is to figure out what needs to be done, how it needs to be done and returns a model. In linear regression, for example, the learning algorithm optimises the cost function and produces the models, i.e. the coefficients.

You just revised the basic machine learning framework. You will now learn a basic property of a learning algorithm - that it can only produce models of a certain kind within its boundaries. This means that an algorithm designed to produce linear class of models, like linear / logistic regression, will never be able to produce a decision tree or a neural network. The class of model becomes critical because a wrong class will yield a sub-optimal model.

![title](image/model3.JPG)

Every learning algorithm puts a boundary around the kinds of models that it is going to ever consider and among those models, it will try to find the best which fits the data that it has been given for training. That model will come out as a output from the learning algorithm. Once you have this model then we can use it to make predicitions.

## Simplicity, Complexity and Overfitting
We will now discuss the notion of model simplicity and complexity in detail and use some examples and analogies to understand the pros and cons of simple and complex models.

![title](image/model-selection.JPG)

Now let's see what kind of learning model to choose, if we choose regression then what kind of regression we are going to use?

![title](image/model-selection1.JPG)

From your school or college, you can probably recall those few fellows who seemed to study less but understood much more than others. They seem to never care about memorizing or mechanically practicing what was being taught, yet are able to explain complex problems in physics or mathematics with simplicity and elegance. 

Assuming that people learn using ‘mental models’, do these students have remarkably different mental models than those who solve a bunch of books and focus on memorization? How can they learn so much from a finite amount of information and apply that to solve unseen, complex problems? 

In this segment, we will explain the meaning of model simplicity, complexity as well as the pros and cons associated with them. As a by-product, you will also understand that the best way to ‘learn’ is ‘to keep your mental models simple’. 

Finally, you learned 4 unique points about using a simpler model where ever possible:

1. A simpler model is usually more generic than a complex model. This becomes important because generic models are bound to perform better on unseen datasets.
2. A simpler model requires less training data points. This becomes extremely important because in many cases one has to work with limited data points.
3. A simple model is more robust and does not change significantly if the training data points undergo small changes.
4. A simple model may make more errors in the training phase but it is bound to outperform complex models when it sees new data. This happens because of overfitting.

### Overfitting
Overfitting is a phenomenon where a model becomes too specific to the data it is trained on and fails to generalise to other unseen data points in the larger domain. A model that has become too specific to a training dataset has actually ‘learnt’ not just the hidden patterns in the data but also the noise and the inconsistencies in the data. In a typical case of overfitting, the model performs very well on the training data but fails miserably on the test data. 

## Bias-Variance Tradeoff
So far, we have discussed the pros and cons of simple and complex models. On one hand, simplicity is generalizable and robust and on the other hand, some problems are inherently complex in nature. There is a trade-off between the two, which is known as the bias-variance tradeoff in machine learning. You will learn about this topic in more detail in the modules to come. 

### Bias and Variance
We considered the example of a model memorizing the entire training dataset. If you change the dataset a little, this model will need to change drastically. The model is, therefore, **unstable and sensitive to changes in training data**, and this is called **high variance**.

![title](image/variance-bias.JPG)

The ‘variance’ of a model is the **variance in its output** on some test data with respect to the changes in the training data. In other words, variance here refers to the **degree of changes in the model itself** with respect to changes in training data.

**Bias** quantifies how accurate the model is likely to be on future (test) data. Extremely simple models are likely to fail in predicting complex real world phenomena. Simplicity has its own disadvantages.

Imagine solving digital image processing problems using simple linear regression when much more complex models like neural networks are typically successful in these problems. We say that the linear model has a high bias since it is way too simple to be able to learn the complexity involved in the task.

In an ideal case, we want to reduce both the bias and the variance, because the expected total error of a model is the sum of the errors in bias and the variance, as shown in the figure below.

![title](image/Bias_variance.png)

Although, in practice, we often cannot have a low bias and low variance model. As the model complexity goes up, the bias reduces while the variance increases, hence the trade-off.