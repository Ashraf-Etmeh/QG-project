'''the class
for generating incorrect alternative
answers for a given answer
'''
from nltk.tokenize import sent_tokenize, word_tokenize
import random
import torch
from nltk.corpus import wordnet as wn
from sentence_transformers import util

from nlp_models import nlp, semantic_model


class IncorrectAnswerGenerator:
    # Class-level caches (shared across ALL instances, survive between requests)
    _embedding_cache_shared = {}
    _wordnet_cache_shared = {}

    def __init__(self, document, doc=None):
        self.document = document
        self.nlp = nlp
        self.ranker_model = semantic_model
        self.doc = doc if doc is not None else self.nlp(document)
        self.document_candidates = self.get_document_candidates()
        # Instance references point to the shared class-level caches
        self._embedding_cache = IncorrectAnswerGenerator._embedding_cache_shared
        self._wordnet_cache = IncorrectAnswerGenerator._wordnet_cache_shared

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

    def get_wordnet_pos_from_span(self, span):
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

        # Return cached result immediately — WordNet traversal is expensive
        cache_key = answer.strip().lower()
        if cache_key in self._wordnet_cache:
            return self._wordnet_cache[cache_key]

        wordnet_candidates = set()

        # Find answer in its original context (once, reused below)
        span = self.find_answer_span(answer)

        # Determine POS from context
        pos_type = self.get_wordnet_pos_from_span(span)

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

        # Cap synset traversal to avoid exploding on answers with large graphs.
        # Top 3 synsets × top 5 hypernyms is more than enough for 3 distractors.
        for syn in synsets[:3]:

            for hypernym in syn.hypernyms()[:5]:

                for hypo in hypernym.hyponyms():

                    for lemma in hypo.lemmas():

                        name = (
                            lemma.name()
                            .replace('_', ' ')
                            .strip()
                        )

                        if name.lower() != answer.lower():
                            wordnet_candidates.add(name)

        result = list(wordnet_candidates)
        self._wordnet_cache[cache_key] = result
        return result

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

    def get_cached_embeddings(self, texts):
        uncached = [t for t in texts if t not in self._embedding_cache]

        if uncached:
            new_embeddings = self.ranker_model.encode(uncached, convert_to_tensor=True)
            for text, embedding in zip(uncached, new_embeddings):
                self._embedding_cache[text] = embedding

        return torch.stack([self._embedding_cache[t] for t in texts])

    def rank_candidates(self, candidates, answer):
        if not candidates:
            return [], None
        # Route the answer through the same cache used for candidates so the
        # transformer is not called again for an answer already encoded in a
        # previous question's distractor pass.
        answer_embedding = self.get_cached_embeddings([answer])[0]
        cand_embeddings = self.get_cached_embeddings(candidates)

        cosine_scores = util.cos_sim(cand_embeddings, answer_embedding)

        order = sorted(
            range(len(candidates)),
            key=lambda i: cosine_scores[i].item(),
            reverse=True
        )

        ranked_candidates = [candidates[i] for i in order]
        ranked_embeddings = cand_embeddings[order]

        return ranked_candidates, ranked_embeddings

    def diversify_candidates(self, ranked_candidates, embeddings, num_options):

        selected = []
        selected_indices = []

        if embeddings is None:
            return selected

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
        ranked, ranked_embeddings = self.rank_candidates(cleaned, answer)
        diversified = self.diversify_candidates(ranked, ranked_embeddings, num_options)

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