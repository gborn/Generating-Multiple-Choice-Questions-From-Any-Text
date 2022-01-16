from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from nltk.tokenize import sent_tokenize
import nltk
# we use nltk library to tokenize our text
nltk.download('punkt')

class AnswerKey:
    """
    Generate answers using keyword extraction, and map them to sentences they appear in
    """

    def __init__(self, text):
        self.text = text

        # KeyBert uses BERT-embeddings and simple cosine similarity to find the sub-phrases in a document that are the most similar to the document itself.
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.kw_model = KeyBERT(sentence_model)

    def get_keywords(self, text):
        """
        Given @input text, identify important keywords. 
        Here we use Sentence Transformer to extract keywords that best describe the text
        """
        keywords_with_scores = self.kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), top_n=5, stop_words='english')
        keywords = [kw[0] for kw in keywords_with_scores]
        scores = [kw[1] for kw in keywords_with_scores]
        return keywords

    def tokenize_sentences(self, text):
        """
        Given a @text input, returns tokenized sentences
        """
        sentences = [sent_tokenize(text)]
        sentences = [sentence for paragraph in sentences for sentence in paragraph]

        # Remove sentences shorter than 20 letters.
        sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
        return sentences

    def get_sentences_for_keyword(self, kw_model, sentences, ngram_range=(1, 1), top_n=10):
        """
        @kw_model: keyBERT model to extract keywords
        @sentences: list of tokenized sentences
        returns a map with keywords as keys mapped to the sentences they appear in.
        """
        keyword_sentences = {}
        for sentence in sentences:
            keywords_found = [kw[0] for kw in kw_model.extract_keywords(sentence, keyphrase_ngram_range=ngram_range, top_n=top_n) if len(kw[0]) > 2]

            for key in keywords_found:
                keyword_sentences[key] = keyword_sentences.get(key, [])
                keyword_sentences[key].append(sentence)

        return keyword_sentences

    def get_answers(self, ngram_range=(1, 2), top_n=10):
        sentences = self.tokenize_sentences(self.text)
        keyword_to_sentences = self.get_sentences_for_keyword(self.kw_model, sentences, ngram_range=ngram_range, top_n=top_n)
        return keyword_to_sentences