
what RQ can I answer: given gyroscopic and accelorometer data, can we determine in who's pocket the phone is?

what have other people done:
- exesiting repo who used CNN and LSTM.


difficulty:
- 98% accuracy, using deep learning (CNN and LSTM)


Idea from WIll:
- want to train the model in a way that it can generalize to new people, so that we can use it in the real world.
- for evaluation: hold out one person, train on the rest, and test on the hold out person.
- unlike the paper who evaluated on the last 10% of a person's data (which measures how well the model can predict on a person it has already seen, but not how well it can generalize to new people).


limitation:
- data may include non-walking activities.


data splitting: 80/10/10 for training/validation/testing, with the hold-out person in the test set.
k-fold cross validation: hold out each person in turn, train on the rest, and test on the hold-out person. This will give us a better estimate of how well the model can generalize to new people. (10-fold, we have 180 people, so we can hold out 18 people in each fold).

preprocessing:
- use a single kernel CNN to detect if phone is walking or not. Then feed only "walking" data into our classifier.

visualization:
- multi-dimensional numerical data: time-series data from the gyroscope and accelerometer. We can visualize this data using line plots, where the x-axis represents time and the y-axis represents the sensor readings. We can also use scatter plots to visualize the relationship between different sensor readings.
- 3D trajectory plots: we can plot the 3D trajectory of the phone's movement using the accelerometer data. This can help us visualize the movement patterns associated with different pocket placements.
- confusion matrix: we can use a confusion matrix to visualize the performance of our classifier. 
- PCA for visualizing the "embeddings" of the data: we can use Principal Component Analysis (PCA) to reduce the dimensionality of our data and visualize it in a 2D or 3D space. This can help us see if there are any clusters or patterns in the data that correspond to different pocket placements.

data preperation:
- data cleaning: sliding window smoothing averaging
- tokenization: IMU tokenization (Inertial Measurement Unit tokenization) to convert the raw sensor data into a format suitable for input into our model. This involves segmenting the time-series data into fixed-length windows and extracting relevant features from each window.

class imbalance:
- lets measure the data set

training strategy (WIll & Tijje)
- K-fold cross validation: hold out each person in turn, train on the rest, and test on the hold-out person. This will give us a better estimate of how well the model can generalize to new people. (10-fold, we have 180 people, so we can hold out 18 people in each fold).
- contrastive learning: learn to embed a walking sequence into a vector space, where the same person's walking sequences are close together, and different people's walking sequences are far apart. This can help the model learn to generalize to new people, as it will learn to focus on the underlying patterns in the data rather than memorizing specific individuals' data. We can use a contrastive loss function, such as triplet loss or contrastive loss, to train the model in this way.


model selection:
- IMO tokenizer
- Baseline: FFT features + SVM or something else
- deep learning models

- loss: triplet loss
- metrics: accuracy, precision, recall, F1 score, confusion matrix, K-fold thingy
- clustering (the embedding of the time-series data): PCA, t-SNE, UMAP
- classification (triplet loss): SVM, KNN, Random Forest, Logistic Regression, Neural Networks (MLP, CNN, LSTM)
- discussion: FP vs FN


overfitting?
- regularization techniques: dropout, L2 regularization, early stopping, batch norm, layer norm, adam
- data regularization: data augmentation (adding noise, time warping, etc.), and using a larger dataset if possible.

hyperparameter tuning:
- grid search: we can perform a grid search over a range of hyperparameters to find the

compress the model:
- pruning: 
- compression: quantization
- data compression:


deployment:
- web app which records the gyroscopic and accelerometer data from the phone, preprocesses it, and feeds it into the trained model to predict which person is carrying the phone.

explainability:
- FFT features: we can use the FFT features to understand which frequencies are most important for distinguishing between different pocket placements. We can visualize the FFT features using a bar plot or a heatmap to see which frequencies are most important for the model's predictions.
- visualize the embedding space: explain what features the model is using to make its predictions by visualizing the embedding space.


non-standard task:
- gait recognition is a non-standard task, as it involves analyzing time-series data from sensors to identify patterns in human movement. This is different from more traditional classification tasks that involve static images or tabular data. However, it has important applications in areas such as security, healthcare, and human-computer interaction.

non-standard method?
- yes, contrastive learning is a non-standard method for this type of problem, but it has been shown to be effective for learning representations that generalize well to new data. By learning to embed the walking sequences into a vector space, we can capture the underlying patterns in the data that are relevant for distinguishing between different pocket placements, rather than relying on memorizing specific individuals' data. This can help the model generalize better to new people and improve its performance on unseen data.

Evaluate with real new data:

- We will record new data from a few volunteers who were not part of the original dataset. We will have them carry the phone in different pockets while walking, and we will use this new data to evaluate the performance of our trained model. This will give us a better understanding of how well our model can generalize to new people and real-world scenarios.


## Internal deadlines

project proposal: 24th of may

<!-- data setup processing: two weeks -->


model deployment: 29th of may 

Model presentation: 10th of june

Reflection: 26th of june



