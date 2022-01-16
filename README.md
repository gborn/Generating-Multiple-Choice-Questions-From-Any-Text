# Generating-Multiple-Choice-Questions-From-Any-Text
A webapp to generate multiple choice questions (MCQs) from any Text. It uses Google's T5 model fined tuned on [SQUAD](https://rajpurkar.github.io/SQuAD-explorer/) dataset to generate questions, and BERT to generate distractors, and answers are found using simple keyword extraction using Sentence Transformer.
