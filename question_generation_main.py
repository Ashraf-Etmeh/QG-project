'''tie together the
questions generation and incorrect answer
generation
'''
from question_extraction import QuestionExtractor
from incorrect_answer_generation import IncorrectAnswerGenerator
from hybrid_assessment_system import HybridAssessmentSystem
import re
from nltk import sent_tokenize


class QuestionGeneration:
    ''' to generate questions '''

    def __init__(self, num_questions, num_options):
        self.num_questions = num_questions
        self.num_options = num_options
        self.question_extractor = QuestionExtractor(num_questions)

    def clean_text(self, text):
        text = text.replace('\n', ' ')

        sentences = sent_tokenize(text)
        cleaned_text = ""

        for sentence in sentences:

            # Remove punctuation (keep letters, numbers and spaces)
            cleaned_sentence = re.sub(r'([^\s\w]|_)+', '', sentence)

            # Remove multiple spaces
            cleaned_sentence = re.sub(r'\s+', ' ', cleaned_sentence).strip()

            # Skip empty sentences
            if not cleaned_sentence:
                continue

            cleaned_text += cleaned_sentence + ". "

        return cleaned_text.strip()

    def generate_questions_dict(self, document):
        document = self.clean_text(document)
        self.questions_dict = self.question_extractor.get_questions_dict(
            document)
        self.incorrect_answer_generator = IncorrectAnswerGenerator(
            document, doc=self.question_extractor.doc)

        for i in range(1, self.num_questions + 1):
            if i not in self.questions_dict:
                continue
            self.questions_dict[i]["options"] \
                = self.incorrect_answer_generator.get_all_options_dict(
                self.questions_dict[i]["answer"],
                self.num_options
            )

        return self.questions_dict


class QuestionGeneration_free:

    def __init__(self, num_questions, num_options):
        self.num_questions = num_questions
        self.num_options = num_options

        # Use the class that actually exists
        self.question_extractor = HybridAssessmentSystem(num_questions)

    def clean_text(self, text):
        text = text.replace("\n", " ")

        sentences = sent_tokenize(text)
        cleaned_sentences = []

        for sentence in sentences:
            cleaned_sentence = re.sub(
                r'([^\s\w]|_)+',
                '',
                sentence
            )

            cleaned_sentence = re.sub(
                r'\s+',
                ' ',
                cleaned_sentence
            ).strip()

            if cleaned_sentence:
                cleaned_sentences.append(cleaned_sentence)

        return ". ".join(cleaned_sentences) + "."

    def generate_questions_dict(self, document):

        document = self.clean_text(document)

        # Use the actual method defined in HybridAssessmentSystem
        self.questions_dict = (
            self.question_extractor.get_assessment(document)
        )

        self.incorrect_answer_generator = (
            IncorrectAnswerGenerator(document, doc=self.question_extractor.doc)
        )

        # Generate options ONLY for MCQs
        for question_num, question_data in self.questions_dict.items():

            if question_data.get("type") != "Multiple-Choice":
                continue

            question_data["options"] = (
                self.incorrect_answer_generator.get_all_options_dict(
                    question_data["answer"],
                    self.num_options
                )
            )

        return self.questions_dict