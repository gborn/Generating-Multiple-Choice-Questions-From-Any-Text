
from transformers import pipeline
from .t5model import query
import random


# load BERT model once
unmasker = pipeline('fill-mask', model='distilbert-base-uncased')


class Model:

    def get_questions(self, keyword_to_sentences_map, model, k=5, declarative=True):
        """"
        Generates questions along with distractors
        @input keyword_to_sentences_map : maps keywords to sentences they appear in
        @model: BERT model that will be used to mask the keyword and generate distractors
        @k (default 5), number of questions to return
        """

        results = []

        # we can choose answer keys randomly from the pool of keywords
        answer_keys = random.choices( list(keyword_to_sentences_map.keys()), k=k)

        for answer in answer_keys:
            sentences = keyword_to_sentences_map[answer]
            sentence = max(sentences, key=len)

            if len(sentence) < 20:
                continue

            start_idx = sentence.lower().find(answer)
            end_idx = start_idx + len(answer)

            # replace answer in sentence with blank line to form question
            question = sentence.replace(sentence[start_idx: end_idx], '__________')

            # generate distractors from BERT model 
            distractors = model(question.replace('__________', '[MASK]'))
            options = [option['token_str'] for option in distractors if isinstance(option, dict) and (answer not in option['token_str'].lower())]
            #print(distractors)

            # generate question
            if not declarative:
                context = sentence
                output = query(f"answer: {answer} context: {context}")
                question_t5 = output[0]["generated_text"]

                if options:
                    results.append((question_t5, options, answer))
            else:
                if options:
                    results.append((question, options, answer))

        return results