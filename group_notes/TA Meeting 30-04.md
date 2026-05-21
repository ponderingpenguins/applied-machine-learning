# TA Meeting 30/04

## Transcription

TA: You want to deploy on device which is cool. But you might consider do it as an extra and have an API deploy assignment. You will receive points for extra. First make API work and then for extra point you can do phone app.

Will: *Something about deployment*

TA: When you deploy on device, how would it work?

Will: Intervals, every 15 seconds embedding, embedding over time, can do as API

TA: Sounds good. You don't have to make it work for device, just have an end-point to send data.

----

Efe: *Shows data analysis*, We have done a t-SNE 

TA: Make sure you have one dimensionality reduction, so this is good. The t-SNE looks nice.

Will: We assume that these are participant groups for the colours (118 classes)

TA: You might consider doing a subset of classes, since it is a bit much. But it is good enough

----

Will: Haukur got 99% accuracy with SVM, we have to look for data leakage or a fluke. But this would be our baseline. Once we know that it works, we want to maybe do a transformer instead of LSTM like the paper. 

Efe: Manually feature selection for the first layer in the transformer.

TA: But do you inted to do both manual feature selection and deep learning.

Will: We wanted to do, deep learning better for quantizing. Will it be more beneficial to do just baseline with the manual feature selection or the baseline with the manual and like the full model with deep learning?

TA: YOu don't have to do deep learning, manual feature selection also works.

Will: I like deep learning, Haukur likes feature selection

Efe: I also like deep learning

TA: You can also do both deep learning and feature selection. Since you are quite ahead on time

----

Will: Would you prefer to have a nicely organized paragraph or bullet points.

TA: I rather have a paragraph. A structure is nice, also because you kinda want to motivate your choices so a paragraph is nicer.

----

Will: Hyperparameter search, is grid search fine?

TA: yeah grid search is pervfect

----

Will: regularizations

TA: Motivate your regularizations and not just throw them at the wall. But it is also important to understand the options.

----

Will: Have you heard about open set ML task?

TA: No

Will: How would you define what our project goal is with the ML task?

TA: To me it feels like a binary classification.

Will: We have written it down as verification

TA: It is not too important for me to give an exact definition, just if the task is clear.

Efe: *gives definition in the proposal*

TA: that sounds fine.

----

Will: How much emphasis is the model usage, would much be on the risk evaluation.

TA: just mention the things in the assignment description. That should be fine. To me it appears you know what you are doing.

----

Will: no point to go through the risk assessment thing

----

TA: We can stop the meeting earlier if you we have said everything.

Will: We plan to write the rest of the proposal. We are gonna cite the papers. How much citation, also motivation? Like triplet loss or just we looked into this

TA: I don't think it is strictly necessary, it is useful for me to look into it. 

----

Will: We fought some merge issues. *Describes the main merge disaster of 30/04*

TA: Good that you experienced this early on :)

----

Will: the lecture talks about the git structure. Will you look at version history much?

TA: No, not much, I will look into it if it is the case. My suggestion, don't overcomplicate it

Efe: We were planning to do what the lecture suggested.

TA: I think that is good. If you are working on different features, it is easier.

Will: Both working on preliminary data with Haukur was a bit of an issue with how we did it.

Will: Notebooks don't work nice when you are reviewing changes. just for preliminary data looking we use notebooks.

----

TA: No other questions, just deploying thing. You know what you are doing, all good.

----

Will: How the other groups doing?

TA: more standard, some maybe a bit too much. Generic things. Biggest issue you are gonna run into is big datasets.

Will: There has to be leakage in our model with 99%.

TA: think so, people walk the same way