from PyPDF2 import PdfReader
from question_generation_main import QuestionGeneration, QuestionGeneration_free
from question_processing import (fix_residual_phrases, clean_options)

def pdf2text(file_path: str, file_exten: str) -> str:
    """ Converts a given file to text content """

    _content = ''

    # Identify file type and get its contents
    if file_exten.lower() == 'pdf':
        with open(file_path, 'rb') as pdf_file:
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    _content += text + "\n"
            # _content = _pdf_reader.getPage(0).extractText()
            print('PDF operation done!')

    elif file_exten == 'txt':
        with open(file_path, 'r') as txt_file:
            _content = txt_file.read()
            print('TXT operation done!')

    return _content


def txt2questions(doc: str, n=5, o=4) -> dict:
    """ Get all questions and options """

    qGen = QuestionGeneration(n, o)
    q = qGen.generate_questions_dict(doc)
    for i in range(len(q)):
        temp = []
        for j in range(len(q[i + 1]['options'])):
            temp.append(q[i + 1]['options'][j + 1])
        # print(temp)
        q[i + 1]['options'] = temp
    return q


def process_new_questions(questions):

    processed_questions = {}

    for q_num, q_data in questions.items():

        q_type = q_data.get(
            "type",
            "Multiple-Choice"
        )

        raw_question = q_data.get(
            "question",
            ""
        )

        answer = q_data.get(
            "answer",
            ""
        )

        clean_question = (
            fix_residual_phrases(
                raw_question,
                answer
            )
        )

        if q_type == "Multiple-Choice":

            options = q_data.get(
                "options",
                []
            )

            options = clean_options(
                options,
                answer,
                max_options=4
            )

            processed_questions[
                q_num
            ] = {

                "type": "Multiple-Choice",

                "question":
                    clean_question,

                "answer":
                    answer,

                "options":
                    options
            }

        elif q_type == "Free-Response":

            processed_questions[
                q_num
            ] = {

                "type": "Free-Response",

                "question":
                    clean_question,

                "answer":
                    answer,

                "options":
                    []
            }

    return processed_questions

def txt2questions_free(doc: str, n=5, o=4) -> dict:
    """Generate questions and convert MCQ options from dict to list."""

    qGen = QuestionGeneration_free(n, o)
    questions = qGen.generate_questions_dict(doc)

    for q_num, q_data in questions.items():

        # Only MCQs have options
        if q_data.get("type") == "Multiple-Choice":
            options = q_data.get("options", {})

            if isinstance(options, dict):
                q_data["options"] = list(options.values())

    process_new_questions(questions)
    return questions