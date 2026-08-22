import random
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

class HybridAssessmentSystem:

  def __init__(self, num_questions):
    self.num_questions = num_questions
    self.stop_words = set(ENGLISH_STOP_WORDS)
    self.ner_tagger = spacy.load("en_core_web_md")
    self.vectorizer = TfidfVectorizer()
    self.assessment_dict = dict()

  def get_assessment(self, document):
    self.candidate_keywords = self.get_candidate_entities(document)
    self.set_tfidf_scores(document)
    self.rank_keywords()
    return self.form_questions()

  def get_filtered_sentences(self, document):
    doc = self.ner_tagger(document)
    sentences = [sent.text.strip() for sent in doc.sents]
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

    doc = self.ner_tagger(document)
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

  def set_tfidf_scores(self, document):
    doc = self.ner_tagger(document)
    self.unfiltered_sentences = [sent.text.strip() for sent in doc.sents]
    self.filtered_sentences = self.get_filtered_sentences(document)

    self.word_score = dict()
    self.sentence_for_max_word_score = dict()

    tf_idf_vector = self.vectorizer.fit_transform(self.filtered_sentences)
    feature_names = self.vectorizer.get_feature_names_out()
    tf_idf_matrix = tf_idf_vector.todense().tolist()

    num_sentences = len(self.unfiltered_sentences)
    num_features = len(feature_names)

    for i in range(num_features):
      word = feature_names[i]
      self.sentence_for_max_word_score[word] = ""
      tot = 0.0
      cur_max = 0.0

      for j in range(num_sentences):
        tot += tf_idf_matrix[j][i]
        if tf_idf_matrix[j][i] > cur_max:
          cur_max = tf_idf_matrix[j][i]
          self.sentence_for_max_word_score[word] = self.unfiltered_sentences[j]

      self.word_score[word] = tot / num_sentences if num_sentences > 0 else 0.0

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

  def generate_clean_interrogative_question(self, sentence, answer):
    doc = self.ner_tagger(sentence)
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
        question_text = self.generate_clean_interrogative_question(
            sentence, raw_answer
        )

        if self.is_valid_question(question_text) and self.validate_semantic_consistency(
            question_text, formatted_answer
        ):
          used_sentences.append(sentence)

          if cntr == 1:
            self.assessment_dict[cntr] = {
                "type": "Free-Response",
                "question": question_text,
                "answer": formatted_answer,
            }
          else:
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