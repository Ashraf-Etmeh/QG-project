import random
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

from nlp_models import nlp

class HybridAssessmentSystem:

  def __init__(self, num_questions):
    self.num_questions = num_questions
    self.stop_words = set(ENGLISH_STOP_WORDS)
    self.ner_tagger = nlp
    self.vectorizer = TfidfVectorizer()
    self.assessment_dict = dict()

  def get_assessment(self, document):
    # parse the document once and reuse it for entities, noun chunks and sentences
    self.doc = self.ner_tagger(document)
    self.candidate_keywords = self.get_candidate_entities(document)
    self.set_tfidf_scores()
    self.rank_keywords()
    return self.form_questions()

  def get_filtered_sentences(self):
    sentences = [sent.text.strip() for sent in self.doc.sents]
    return [self.filter_sentence(sentence) for sentence in sentences]

  def filter_sentence(self, sentence):
    words = sentence.split()
    return " ".join(w for w in words if w.lower() not in self.stop_words)

  def get_candidate_entities(self, document):
    entity_list = []
    time_patterns = re.findall(
        r"\b(?:[0-9]+|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|twelve|thirty))\s+(?:consecutive\s+)?(?:day|days|week|weeks|month|months|year|years|hour|hours)\b",
        document,
        re.IGNORECASE,
    )
    for match in time_patterns:
      cleaned = re.sub(r"\s+", " ", match).strip()
      if len(cleaned) > 2:
        entity_list.append(cleaned)

    doc = self.doc
    for ent in doc.ents:
      text = ent.text.strip()
      if (
          len(text.split()) <= 4
          and len(text) > 2
          and not text.endswith((" Or", " And", " The", " A"))
      ):
        entity_list.append(text)

    for chunk in doc.noun_chunks:
      text = chunk.text.strip()
      text = re.sub(
          r"^(the|a|an|its|their|our|my|your)\s+", "", text, flags=re.IGNORECASE
      )
      if (
          2 <= len(text.split()) <= 4
          and len(text) < 30
          and not text.lower().endswith((" or", " and", " of", " in", " to"))
          and text not in entity_list
      ):
        entity_list.append(text)

    return list(dict.fromkeys(entity_list))

  def set_tfidf_scores(self):
    self.unfiltered_sentences = [sent.text.strip() for sent in self.doc.sents]
    self.filtered_sentences = self.get_filtered_sentences()

    self.word_score = dict()
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
    # Convert to CSR for row-efficient argmax per column via transpose trick.
    tf_idf_csr = tf_idf_csc.tocsr()
    # argmax over rows for each column: operate on the transpose (rows=features)
    col_argmax = tf_idf_csr.T.argmax(axis=1).A1   # shape (num_features,)

    for i, word in enumerate(feature_names):
      self.word_score[word] = float(avg_scores[i])
      self.sentence_for_max_word_score[word] = (
          self.unfiltered_sentences[col_argmax[i]]
          if avg_scores[i] > 0.0
          else ""
      )

  def get_keyword_score(self, keyword):
    score = 0.0
    for word in keyword.split():
      if word in self.word_score:
        score += self.word_score[word]
    return score

  def get_corresponding_sentence_for_keyword(self, keyword):
    keyword_tokens = keyword.lower().split()
    for word in keyword_tokens:
      if word not in self.sentence_for_max_word_score:
        continue
      sentence = self.sentence_for_max_word_score[word]
      sentence_lower = sentence.lower()

      all_present = True
      for kw in keyword_tokens:
        pattern = r"\b" + re.escape(kw) + r"\b"
        if not re.search(pattern, sentence_lower):
          all_present = False
          break

      if all_present and len(sentence.split()) > 5:
        return sentence
    return ""

  def rank_keywords(self):
    self.candidate_triples = []
    for candidate_keyword in self.candidate_keywords:
      self.candidate_triples.append([
          self.get_keyword_score(candidate_keyword),
          candidate_keyword,
          self.get_corresponding_sentence_for_keyword(candidate_keyword),
      ])
    self.candidate_triples.sort(reverse=True)

  def is_valid_question(self, question):
    q_lower = question.lower()
    words = question.split()
    if len(words) < 4 or len(words) > 25:
      return False
    if not (question.endswith("?") or "______" in question):
      return False

    valid_starters = (
        "how",
        "what",
        "who",
        "where",
        "when",
        "why",
        "is",
        "are",
        "can",
        "do",
        "does",
        "which",
    )
    if not q_lower.startswith(valid_starters):
      return False
    if "can can" in q_lower or "do does" in q_lower or "  " in question:
      return False
    return True

  def validate_semantic_consistency(self, question, answer):
    q_lower = question.lower()
    ans_lower = answer.lower()
    is_time_answer = any(
        w in ans_lower
        for w in [
            "day",
            "days",
            "week",
            "weeks",
            "month",
            "months",
            "year",
            "years",
            "hour",
            "hours",
        ]
    )
    is_person_answer = any(
        w in ans_lower
        for w in [
            "employee",
            "employees",
            "staff",
            "personnel",
            "worker",
            "director",
            "manager",
        ]
    )

    if is_time_answer and not q_lower.startswith("how long"):
      return False
    if is_person_answer and q_lower.startswith("how long"):
      return False
    return True

  
  # Free-Response helpers

  def rank_sentences_for_free_response(self):
    """Return unfiltered sentences sorted by their aggregate TF-IDF score.

    Reuses the word_score and sentence_for_max_word_score data already
    populated by set_tfidf_scores().  No new vectorisation is performed.
    """
    scored = []
    for idx, sent in enumerate(self.unfiltered_sentences):
      filtered = self.filtered_sentences[idx]
      tokens = filtered.split()
      score = sum(self.word_score.get(t.lower(), 0.0) for t in tokens)
      # Prefer sentences of a reasonable length (not one-liners, not walls)
      word_count = len(sent.split())
      if word_count < 6 or word_count > 60:
        score *= 0.5
      scored.append((score, idx, sent))
    scored.sort(reverse=True)
    return scored  # list of (score, original_index, sentence_text)

  def extract_topic_from_sentence(self, sentence_doc):
    """Return a short topic phrase that describes what the sentence is about.

    Accepts a pre-parsed spaCy Doc or Span (from self.doc.sents) so the
    sentence is never re-parsed.  Falls back gracefully when the parse is
    sparse.

    Strategy (in priority order):
      1. The noun-chunk that is the grammatical subject (nsubj / nsubjpass).
      2. The first named entity.
      3. The first noun chunk.
      4. The first two content words.
    """
    doc = sentence_doc

    # 1. Subject noun chunk
    for chunk in doc.noun_chunks:
      if chunk.root.dep_ in ("nsubj", "nsubjpass"):
        text = re.sub(
            r"^(the|a|an|its|their|our|my|your)\s+",
            "",
            chunk.text.strip(),
            flags=re.IGNORECASE,
        )
        if text:
          return text.strip()

    # 2. First named entity
    for ent in doc.ents:
      if len(ent.text.strip()) > 2:
        return ent.text.strip()

    # 3. First noun chunk
    for chunk in doc.noun_chunks:
      text = re.sub(
          r"^(the|a|an|its|their|our|my|your)\s+",
          "",
          chunk.text.strip(),
          flags=re.IGNORECASE,
      )
      if text:
        return text.strip()

    # 4. Fallback: first two non-stop content words
    content = [
        t.text for t in doc
        if not t.is_stop and not t.is_punct and t.pos_ in ("NOUN", "PROPN", "VERB")
    ]
    return " ".join(content[:2]) if content else ""

  def generate_free_response_question(self, sentence, sentence_doc=None):
    """Generate an open-ended question whose answer is the full sentence.

    Accepts an optional pre-parsed spaCy Doc/Span (sentence_doc) so the
    sentence is never re-parsed.  Falls back to parsing on-demand only when
    no pre-parsed span is supplied.

    Returns (question_text, reference_answer) where reference_answer is
    the original unfiltered sentence.
    """
    doc = sentence_doc if sentence_doc is not None else self.ner_tagger(sentence)
    topic = self.extract_topic_from_sentence(doc)

    # Identify the root verb to choose an appropriate question frame
    root_verb = None
    root_token = None
    for token in doc:
      if token.dep_ == "ROOT":
        root_token = token
        root_verb = token.lemma_.lower()
        break

    # Collect the direct object or complement to the root verb (used for
    # choosing a more precise question word)
    dobj_text = ""
    for token in doc:
      if token.dep_ in ("dobj", "attr", "acomp") and token.head == root_token:
        dobj_text = token.text.lower()
        break

    #     Detect sentence semantics to pick the right question frame 

    sent_lower = sentence.lower()

    # Obligation / requirement / policy sentences
    obligation_markers = (
        "must", "shall", "required", "requirement", "responsible",
        "obligation", "obligated", "expected to", "need to",
        "prohibited", "forbidden", "not allowed", "policy",
    )
    is_obligation = any(m in sent_lower for m in obligation_markers)

    # Process / procedure sentences
    process_markers = (
        "submit", "request", "apply", "approve", "review",
        "complete", "provide", "report", "notify", "ensure",
        "conduct", "perform", "follow",
    )
    is_process = any(m in sent_lower for m in process_markers)

    # Temporal / deadline sentences
    time_markers = (
        "days", "weeks", "months", "years", "hours",
        "deadline", "prior to", "before", "within", "no later than",
        "at least", "no more than",
    )
    is_temporal = any(m in sent_lower for m in time_markers)

    # Role / responsibility sentences
    role_markers = (
        "responsible for", "in charge of", "accountable",
        "oversees", "manages", "handles", "leads",
    )
    is_role = any(m in sent_lower for m in role_markers)

    #    Build the question 
    topic_lower = topic.lower() if topic else ""

    if is_role and topic:
      question = f"What responsibility or role do {topic_lower} have regarding this matter?"
      # Refine: replace "this matter" with the dobj if available
      if dobj_text:
        question = f"What responsibility do {topic_lower} have regarding {dobj_text}?"
      else:
        # Try to pull in the prepositional phrase after the root verb
        prep_phrases = [
            chunk.text for chunk in doc.noun_chunks
            if chunk.root.dep_ == "pobj"
        ]
        if prep_phrases:
          question = (
              f"What is the role or responsibility of {topic_lower} "
              f"concerning {prep_phrases[0].lower()}?"
          )

    elif is_obligation and is_temporal and topic:
      question = (
          f"What time-related requirement applies to {topic_lower} "
          f"under this policy?"
      )

    elif is_obligation and topic:
      question = f"What requirement or obligation applies to {topic_lower}?"

    elif is_process and topic:
      question = f"What must {topic_lower} do according to this policy?"

    elif root_verb in ("be", "is", "are", "was", "were") and topic:
      question = f"What is stated about {topic_lower} in this policy?"

    elif topic:
      question = f"What does the policy specify about {topic_lower}?"

    else:
      question = "What does this policy provision state?"

    # Capitalise and ensure question mark
    question = re.sub(r"\s+", " ", question).strip()
    if not question.endswith("?"):
      question += "?"
    question = question[0].upper() + question[1:]

    return question, sentence.strip()

  def is_valid_free_response_question(self, question):
    """Lightweight validation specific to Free-Response questions.

    Intentionally more permissive than is_valid_question() because FR
    questions are allowed to be longer and do not need 'options'.
    The MCQ is_valid_question() is NOT modified.
    """
    words = question.split()
    if len(words) < 5 or len(words) > 35:
      return False
    if not question.strip().endswith("?"):
      return False
    q_lower = question.lower()
    valid_starters = (
        "what", "how", "who", "where", "when", "why",
        "which", "describe", "explain",
    )
    if not q_lower.startswith(valid_starters):
      return False
    # Reject obviously vague questions
    vague_patterns = (
        "what does the policy state regarding this matter",
        "what is stated in this policy",
    )
    if any(p in q_lower for p in vague_patterns):
      return False
    return True


  def generate_clean_interrogative_question(self, sentence, answer, sentence_doc=None):
    doc = sentence_doc if sentence_doc is not None else self.ner_tagger(sentence)
    clean_ans = re.sub(r"\s+", " ", answer).strip()
    clean_ans_lower = clean_ans.lower()

    answer_tokens = [
        token for token in doc if token.text.lower() in clean_ans_lower
    ]
    dep_role = ""
    for token in answer_tokens:
      if token.dep_ in ("nsubj", "nsubjpass"):
        dep_role = "subject"
        break
      elif token.dep_ in ("dobj", "pobj", "attr"):
        dep_role = "object"
        break

    temp_sent = sentence
    for prep in [r"\bup to\b", r"\bfor\b", r"\bduring\b", r"\bin\b", r"\bat\b"]:
      temp_sent = re.sub(
          prep + r"\s+" + re.escape(clean_ans),
          "",
          temp_sent,
          flags=re.IGNORECASE,
      )

    temp_sent = re.sub(re.escape(clean_ans), "", temp_sent, flags=re.IGNORECASE)
    temp_sent = re.sub(r"\s+", " ", temp_sent).strip(" ,.-?")

    words = temp_sent.split()
    deduped_words = [
        w
        for i, w in enumerate(words)
        if i == 0 or w.lower() != words[i - 1].lower()
    ]
    temp_sent = " ".join(deduped_words[:12])

    is_time_answer = any(
        w in clean_ans_lower
        for w in [
            "day",
            "days",
            "week",
            "weeks",
            "month",
            "months",
            "year",
            "years",
            "hour",
            "hours",
        ]
    )
    is_person_answer = any(
        w in clean_ans_lower
        for w in [
            "employee",
            "employees",
            "staff",
            "personnel",
            "worker",
            "director",
            "manager",
        ]
    )

    if is_time_answer:
      question = f"How long is specified regarding {temp_sent.lower()}?"
    elif is_person_answer and dep_role == "subject":
      question = f"Who is responsible for or applies to {temp_sent.lower()}?"
    elif is_person_answer:
      question = (
          f"Which category or group of personnel applies to {temp_sent.lower()}?"
      )
    else:
      question = (
          f"What is the applicable policy or provision regarding"
          f" {temp_sent.lower()}?"
      )

    return re.sub(r"\s+", " ", question).strip().capitalize()

  def form_questions(self):
    used_sentences = list()
    idx = 0
    cntr = 1
    num_candidates = len(self.candidate_triples)
    all_answers = [
        re.sub(r"\s+", " ", trip[1]).strip().title()
        for trip in self.candidate_triples
        if trip[1]
    ]

    # Build a lookup from sentence text → spaCy Span once so every
    # downstream call can reuse the already-parsed span instead of
    # calling self.ner_tagger(sentence) again.
    sent_span_lookup = {span.text.strip(): span for span in self.doc.sents}

    # Question #1: sentence-based Free-Response              
    # Pick the highest-scoring sentence not used yet, try each in 
    # descending TF-IDF order until a valid question is produced.   

    ranked_sentences = self.rank_sentences_for_free_response()
    fr_generated = False
    for _score, _orig_idx, fr_candidate_sentence in ranked_sentences:
      if fr_candidate_sentence in used_sentences:
        continue
      fr_span = sent_span_lookup.get(fr_candidate_sentence)
      fr_question_text, fr_reference_answer = self.generate_free_response_question(
          fr_candidate_sentence, sentence_doc=fr_span
      )
      if self.is_valid_free_response_question(fr_question_text):
        used_sentences.append(fr_candidate_sentence)
        self.assessment_dict[1] = {
            "type": "Free-Response",
            "question": fr_question_text,
            "answer": fr_reference_answer,
        }
        cntr = 2          # FR question done; MCQ loop starts at 2
        fr_generated = True
        break

    if not fr_generated:
      # Absolute fallback: use the first ranked sentence verbatim
      if ranked_sentences:
        _s, _i, fb_sentence = ranked_sentences[0]
        used_sentences.append(fb_sentence)
        self.assessment_dict[1] = {
            "type": "Free-Response",
            "question": "What does this policy provision state?",
            "answer": fb_sentence.strip(),
        }
      cntr = 2

    # Questions #2..N: MCQ path 
    while cntr <= self.num_questions and idx < num_candidates:
      candidate_triple = self.candidate_triples[idx]
      sentence = candidate_triple[2]
      raw_answer = candidate_triple[1]

      if (
          sentence
          and sentence not in used_sentences
          and not raw_answer.lower().endswith((" or", " and", " of", " the"))
      ):
        formatted_answer = re.sub(r"\s+", " ", raw_answer).strip().title()
        sent_span = sent_span_lookup.get(sentence)
        question_text = self.generate_clean_interrogative_question(
            sentence, raw_answer, sentence_doc=sent_span
        )

        if self.is_valid_question(question_text) and self.validate_semantic_consistency(
            question_text, formatted_answer
        ):
          used_sentences.append(sentence)

          distractors = [ans for ans in all_answers if ans != formatted_answer]
          selected_distractors = random.sample(
              distractors, min(3, len(distractors))
          )
          while len(selected_distractors) < 3:
            selected_distractors.append(f"Alternative Option {cntr}")

          options = [formatted_answer] + selected_distractors
          random.shuffle(options)

          self.assessment_dict[cntr] = {
              "type": "Multiple-Choice",
              "question": question_text,
              "answer": formatted_answer,
              "options": options,
          }
          cntr += 1
      idx += 1
    return self.assessment_dict