# cultural-qa-benchmark
Welcome to the Behind the Secrets of Large Language Models Project Page

This benchmark is designed to assess the accuracy of the model you build and to compare your results with those of your peers.
The training data contains English-language questions originating from four different cultural contexts.

There are two tasks included in the benchmark:

Short Answer Questions (SAQ)
Multiple Choice Questions (MCQ)
We begin with the simpler MCQ task. You are presented with a question and four possible answers, labeled A through D. The model must interpret cultural and linguistic cues to select the most appropriate option.

Example (MCQ):
What is the most popular traditional musical instrument in the UK? Choose only one option (A–D).

A. angklung
B. derbouka
C. erhu
D. guitar

Correct answer: D

Next is the SAQ task, which extends the MCQ task. Instead of selecting from predefined options, your model must generate the answer directly.

Example (SAQ):
On which holiday do all family members tend to reunite in the US?

Acceptable answers:

thanksgiving
christmas
Model performance for the MCQ task is evaluated using accuracy, computed through a one-to-one comparison between your predicted answer and the answer key. Since each question has exactly one correct option, accuracy reflects the proportion of correct predictions.

For the SAQ task, accuracy is calculated based on human-annotated answers. A model-generated response is considered correct if it matches any of the provided annotations. Note that this does not account for synonyms or paraphrasing, so keep this limitation in mind.

Most importantly, have fun exploring and experimenting with your model!
