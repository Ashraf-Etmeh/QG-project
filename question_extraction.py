'''the module for generating
'''
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

from nlp_models import nlp


class QuestionExtractor:
    ''' This class contains all the methods
    required for extracting questions from
    a given document
    '''

    def __init__(self, num_questions):

        self.num_questions = num_questions

        # hash set for fast lookup
        self.stop_words = set(stopwords.words('english'))

        # named entity recognition tagger
        self.ner_tagger = nlp

        self.vectorizer = TfidfVectorizer()

        self.questions_dict = dict()

    def get_questions_dict(self, document):
        '''
        Returns a dict of questions in the format:
        question_number: {
            question: str
            answer: str
        }

        Params:
            * document : string
        Returns:
            * dict
        '''
        # parse the document once and reuse it for entity/noun-chunk extraction
        self.doc = self.ner_tagger(document)

        # find candidate keywords
        self.candidate_keywords = self.get_candidate_entities(document)

        # set word scores before ranking candidate keywords
        self.set_tfidf_scores(document)

        # rank the keywords using calculated tf idf scores
        self.rank_keywords()

        # form the questions
        self.form_questions()

        return self.questions_dict

    def get_filtered_sentences(self, document):
        ''' Returns a list of sentences - each of
        which has been cleaned of stopwords.
        Params:
                * document: a paragraph of sentences
        Returns:
                * list<str> : list of string
        '''
        sentences = sent_tokenize(document)  # split documents into sentences

        return [self.filter_sentence(sentence) for sentence in sentences]

    def filter_sentence(self, sentence):
        '''Returns the sentence without stopwords
        Params:
                * sentence: A string
        Returns:
                * string
        '''
        words = word_tokenize(sentence)
        return ' '.join(w for w in words if w not in self.stop_words)

    def get_candidate_entities(self, document):
        ''' Returns a list of filtered entities (shorter and meaningful keywords)
        '''
        entity_list = []

        for ent in self.doc.ents:
            text = ent.text.strip()
            # شروط التنقية: أن تكون الكلمة المفتاحية قصيرة (أقل من 30 حرفاً) ولا تحتوي على مسافات كثيرة
            if len(text.split()) <= 4 and len(text) < 30:
                entity_list.append(text)

        # إذا لم تجد الكيانات ما يكفي، يمكننا أخذ الأسماء (Nouns) أو العبارات القصيرة
        if len(entity_list) < self.num_questions:
            for chunk in self.doc.noun_chunks:
                text = chunk.text.strip()
                if len(text.split()) <= 3 and len(text) < 25 and text not in entity_list:
                    entity_list.append(text)

        return list(set(entity_list))

    def set_tfidf_scores(self, document):
        ''' Sets the tf-idf scores for each word'''
        self.unfiltered_sentences = sent_tokenize(document)
        self.filtered_sentences = self.get_filtered_sentences(document)

        self.word_score = dict()  # (word, score)

        # (word, sentence where word score is max)
        self.sentence_for_max_word_score = dict()

        tf_idf_vector = self.vectorizer.fit_transform(self.filtered_sentences)
        feature_names = self.vectorizer.get_feature_names_out()

        num_sentences = len(self.unfiltered_sentences)
        if num_sentences == 0:
            return

        # Keep the matrix sparse (CSC layout gives fast column slicing).
        # All aggregates are computed with numpy/scipy array ops — no Python
        # loops over individual cells.
        tf_idf_csc = tf_idf_vector.tocsc()

        # Per-feature column sum → average score
        col_sums = tf_idf_csc.sum(axis=0).A1          # shape (num_features,)
        avg_scores = col_sums / num_sentences

        # Per-feature argmax row index (sentence with highest score for that word)
        tf_idf_csr = tf_idf_csc.tocsr()
        col_argmax = tf_idf_csr.T.argmax(axis=1).A1   # shape (num_features,)

        for i, word in enumerate(feature_names):
            self.word_score[word] = float(avg_scores[i])
            self.sentence_for_max_word_score[word] = (
                self.unfiltered_sentences[col_argmax[i]]
                if avg_scores[i] > 0.0
                else ""
            )

    def get_keyword_score(self, keyword):
        ''' Returns the score for a keyword
        Params:
            * keyword : string of possible several words
        Returns:
            * float : score
        '''
        score = 0.0
        for word in word_tokenize(keyword):
            if word in self.word_score:
                score += self.word_score[word]
        return score

    def get_corresponding_sentence_for_keyword(self, keyword):
        ''' Finds and returns a sentence containing
        the keywords using precise token matching (Word Boundaries)
        '''
        keyword_tokens = word_tokenize(keyword.lower())

        for word in keyword_tokens:
            if word not in self.sentence_for_max_word_score:
                continue

            sentence = self.sentence_for_max_word_score[word]
            sentence_lower = sentence.lower()

            all_present = True
            for kw in keyword_tokens:
                pattern = r'\b' + re.escape(kw) + r'\b'
                if not re.search(pattern, sentence_lower):
                    all_present = False
                    break

            if all_present and len(sentence.split()) > 5:
                return sentence

        return ""

    def rank_keywords(self):
        '''Rank keywords according to their score'''
        self.candidate_triples = []  # (score, keyword, corresponding sentence)

        for candidate_keyword in self.candidate_keywords:
            self.candidate_triples.append([
                self.get_keyword_score(candidate_keyword),
                candidate_keyword,
                self.get_corresponding_sentence_for_keyword(candidate_keyword)
            ])

        self.candidate_triples.sort(reverse=True)

    def form_questions(self):
        ''' Forms the question and populates the question dict '''
        used_sentences = list()
        idx = 0
        cntr = 1
        num_candidates = len(self.candidate_triples)

        while cntr <= self.num_questions and idx < num_candidates:
            candidate_triple = self.candidate_triples[idx]
            sentence = candidate_triple[2]
            raw_answer = candidate_triple[1]

            if sentence and sentence not in used_sentences:
                used_sentences.append(sentence)

                pattern = re.compile(r'\b' + re.escape(raw_answer) + r'\b', re.IGNORECASE)

                blank = " ____ "
                question_text = pattern.sub(blank, sentence, count=1)

                formatted_answer = raw_answer.title()

                self.questions_dict[cntr] = {
                    "question": question_text,
                    "answer": formatted_answer
                }

                cntr += 1
            idx += 1
