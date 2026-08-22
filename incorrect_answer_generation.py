'''the class
for generating incorrect alternative
answers for a given answer
'''
from nltk.tokenize import sent_tokenize, word_tokenize
import random
import numpy as np
import spacy
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer, util


class IncorrectAnswerGenerator:
    def __init__(self, document):
        self.document = document
        self.nlp = spacy.load('en_core_web_md')
        self.ranker_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.doc = self.nlp(document)
        self.document_candidates = self.get_document_candidates()

    def get_document_candidates(self):
        doc = self.doc
        candidates = set()
        for chunk in doc.noun_chunks:
            text = chunk.text.strip().title()
            if 2 <= len(text.split()) <= 4:
                candidates.add(text)
        for ent in doc.ents:
            text = ent.text.strip().title()
            if 1 <= len(text.split()) <= 4:
                candidates.add(text)
        return list(candidates)

    def find_answer_span(self, answer):
        """
        Find the answer in the original document and return
        the spaCy span together with its containing sentence.
        """

        answer_tokens = answer.lower().split()

        for i in range(len(self.doc) - len(answer_tokens) + 1):

            span = self.doc[i:i + len(answer_tokens)]

            span_tokens = [
                token.text.lower()
                for token in span
            ]

            if span_tokens == answer_tokens:
                return span

        return None

    def get_wordnet_pos_from_document(self, answer):
        span = self.find_answer_span(answer)

        if span is None:
            return None

        root = span.root

        if root.pos_ == "NOUN":
            return wn.NOUN

        elif root.pos_ == "VERB":
            return wn.VERB

        elif root.pos_ == "ADJ":
            return wn.ADJ

        elif root.pos_ == "ADV":
            return wn.ADV

        return None

    def get_wordnet_candidates(self, answer):

        wordnet_candidates = set()

        # Find answer in its original context
        span = self.find_answer_span(answer)

        # Determine POS from context
        pos_type = self.get_wordnet_pos_from_document(answer)

        answer_key = answer.lower().replace(" ", "_")

        if pos_type:
            synsets = wn.synsets(answer_key, pos=pos_type)
        else:
            synsets = wn.synsets(answer_key)

        if not synsets and span is not None:

            root = span.root

            root_lemma = root.lemma_.lower()

            if pos_type:
                synsets = wn.synsets(
                    root_lemma,
                    pos=pos_type
                )
            else:
                synsets = wn.synsets(root_lemma)

        for syn in synsets:

            for hypernym in syn.hypernyms():

                for hypo in hypernym.hyponyms():

                    for lemma in hypo.lemmas():

                        name = (
                            lemma.name()
                            .replace('_', ' ')
                            .strip()
                        )

                        if name.lower() != answer.lower():
                            wordnet_candidates.add(name)

        return list(wordnet_candidates)

    def filter_candidates(self, candidates, answer):
        filtered = []

        answer_lower = answer.strip().lower()
        answer_is_numeric = any(char.isdigit() for char in answer_lower)

        for cand in candidates:

            cand = cand.strip()

            if not cand: 
                continue

            cand_lower = cand.lower()

            if cand_lower == answer_lower or answer_lower in cand_lower or cand_lower in answer_lower:
                continue

            cand_is_numeric = any(char.isdigit() for char in cand_lower)

            if answer_is_numeric != cand_is_numeric:
                continue

            if cand not in filtered:
                filtered.append(cand)

        return filtered

    def rank_candidates(self, candidates, answer):
        if not candidates:
            return []
        answer_embedding = self.ranker_model.encode(answer, convert_to_tensor=True)
        cand_embeddings = self.ranker_model.encode(candidates, convert_to_tensor=True)

        cosine_scores = util.cos_sim(cand_embeddings, answer_embedding)

        scored_candidates = []
        for i, score in enumerate(cosine_scores):
            scored_candidates.append((score.item(), candidates[i]))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return [cand[1] for cand in scored_candidates]

    def diversify_candidates(self, ranked_candidates, num_options):

        selected = []
        selected_indices = []

        embeddings = self.ranker_model.encode(
            ranked_candidates,
            convert_to_tensor=True,
            show_progress_bar=False
        )

        for i, cand in enumerate(ranked_candidates):

            if len(selected) >= num_options - 1:
                break

            if not selected_indices:
                selected.append(cand)
                selected_indices.append(i)
                continue

            # Get all selected embeddings at once
            selected_embeddings = embeddings[selected_indices]

            # Compare current candidate with ALL selected candidates
            similarities = util.cos_sim(
                embeddings[i],
                selected_embeddings
            )[0]

            # If candidate is too similar to ANY selected candidate
            if similarities.max().item() <= 0.65:
                selected.append(cand)
                selected_indices.append(i)

        return selected

    def get_all_options_dict(self, answer, num_options):
        doc_cands = self.document_candidates
        wn_cands = self.get_wordnet_candidates(answer)
        all_cands = list(set(doc_cands + wn_cands))

        cleaned = self.filter_candidates(all_cands, answer)
        ranked = self.rank_candidates(cleaned, answer)
        diversified = self.diversify_candidates(ranked, num_options)

        if len(diversified) < num_options - 1:
            for cand in ranked:
                if cand not in diversified and cand.lower() != answer.lower():
                    diversified.append(cand)
                if len(diversified) >= num_options - 1:
                    break

        final_distractors = diversified[:num_options - 1]

        options_list = final_distractors + [answer]
        random.shuffle(options_list)

        options_dict = {}
        for i, opt in enumerate(options_list, 1):
            options_dict[i] = opt

        return options_dict