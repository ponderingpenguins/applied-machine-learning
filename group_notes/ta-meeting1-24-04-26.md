
# TA meeting - 24.04.26 at 10:30

members: Will, Tijje, Haukur, Efe, TA: Sebastian Pusch, mail: s.pusch@student.rug.nl 

For Haukur's notes from this transcription, see [here](./ta-meeting1-notes.md).


## Transcription


Will described the project to the TA

- TA: I like the project, its easy to justify why it is useful, which is good


TA: How is the model trained, what loss function, etc.?

- Will: Described the training process, and the loss function, and the evaluation metrics


TA: What compression method will you use?

- will: Pruning, compression...
- Haukur: Quantization


will: What is a good first step for compression?

- TA: Not familiar, he thinks, trying to dive into the literature, and send it to him if it is feasible


Haukur: Does the project fit the scope of the course?

- TA: Yes, it is a good project, and it is in the scope of the course. Especially, because it has a nice novelty aspect


Haukur: Should we compress the embeddings?

- TA: Its an ok, idea, but remember the bottle neck is the model itself.


TA: [[missing question]]

- Will: Class balancing: should can we train on 80% of the people, and then test on the remaining 20%?
- TA: Did not answer question, asked how the model is trained


Haukur: Should we evaluate our model the same way as the original paper?

- TA: Yes, it is a good idea, but you can also do some other evaluation


Haukur: Is our baseline (FFT + SVM) a good baseline, or should we replicate the original paper?

- TA: It is a good baseline, but you can also replicate the original paper, and compare the results


Will: Would be nice to present with a phone "on stage"

- TA: You are not graded on the presentation, but it is a good idea to show the phone. would be cool, but it is risky
- Haukur: Is a video a good idea for the presentation?
- TA: Yes that is a good idea, and a safe option.


Will: Would you recoomend we propose to have a working model <plus something more>?

- TA: Yes, should have a working model, and put the on-edge model as a deployment step (involving compression, etc.)


Haukur: Can you look at our timeline, and give us feedback on it?

- TA: [[looks at timeline]] it looks ambitous, but you have room to extend it a bit, so after week 3 or 4 you can adjust the deadline. If you think training is realistic, its a good timeline


Haukur: What were your problems last year in this course?

- TA: Advice, identify the main bottlenecks, or complexities, and try to solve them early on. Example: compression. Make sure you have given items the right weight, especially things you are doing for the first time.


Will: We have a good repo, we have a CI pipeline. what are your recommendations on the repo?

- TA: Suggests, if CI is extra points, keep it as simple as possible, because if not, you have spent extra time on unnecessary things. CI might slow you down. Recommends simple: linters.


Will: Should we do deployment instead of CI?

- TA: Do you really want to spend time setting up complex CI/CD pipelines? It would be nice, but he feels like it might not be worth it.


Haukur: What are the easy wins?

- TA: project management, github issues, docker, project management tools.


TA: I will send availability, then we can schedule meetings, for every week?

- us: Yes, sounds good!


Meeting ended at 11:10 (40 minutes)
