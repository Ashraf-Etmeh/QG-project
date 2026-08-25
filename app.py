import os
from flask import Flask, render_template, redirect, url_for, request, session
from werkzeug.utils import secure_filename
from workers import pdf2text, txt2questions, txt2questions_free
from question_processing import evaluate_free_response

UPLOAD_FOLDER = './pdf/'

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = '20032026FAG'


@ app.route('/')
def index():
    """ The landing page for the app """
    return render_template('index.html')


@ app.route('/quiz', methods=['GET', 'POST'])
def quiz():
    """ Handle upload and conversion of file + other stuff """

    UPLOAD_STATUS = False
    questions = dict()

    # Make directory to store uploaded files, if not exists
    if not os.path.isdir('./pdf'):
        os.mkdir('./pdf')

    if request.method == 'POST':
        try:
            # Retrieve file from request
            uploaded_file = request.files['file']
            file_path = os.path.join(
                app.config['UPLOAD_FOLDER'],
                secure_filename(
                    uploaded_file.filename))
            file_exten = uploaded_file.filename.rsplit('.', 1)[1].lower()

            uploaded_file.save(file_path)
            # Get contents of file
            uploaded_content = pdf2text(file_path, file_exten)
            # choose question generator
            generation_method = request.form.get('generation_method', 'blank_based')

            if generation_method == 'blank_based':
                questions = txt2questions(uploaded_content)
                for question_id, question_data in questions.items():
                    question_data['type'] = 'Multiple-Choice'

            elif generation_method == 'free_based':
                questions = txt2questions_free(uploaded_content)
                # questions = process_new_questions(questions)

            else:
                raise ValueError(
                    f'Unknown generator: {generation_method}'
                )

            # Store questions for /result
            session['questions'] = questions

            session['generator'] = generation_method

            # File upload + convert success
            if uploaded_content is not None:
                UPLOAD_STATUS = True
        except Exception as e:
            print(e)
    return render_template(
        'quiz.html',
        uploaded=UPLOAD_STATUS,
        questions=questions,
        size=len(questions))


@app.route('/result', methods=['POST'])
def result():

    questions = session.get('questions', {})

    if not questions:
        return "No active quiz.", 400

    correct_q = 0
    total = len(questions)

    review_data = {}

    for question_id, question_data in questions.items():

        q_type = question_data.get(
            'type',
            'Multiple-Choice'
        )

        correct_answer = question_data.get(
            'answer',
            ''
        )

        user_answer = request.form.get(
            str(question_id),
            ''
        )

        if q_type == 'Multiple-Choice':

            is_correct = (
                user_answer.strip().casefold()
                == correct_answer.strip().casefold()
            )

            if is_correct:
                correct_q += 1

            review_data[question_id] = {

                'question': question_data.get(
                    'question',
                    ''
                ),

                'type': 'Multiple-Choice',

                'options': question_data.get(
                    'options',
                    []
                ),

                'user_answer': user_answer,

                'correct_answer': correct_answer,

                'correct': is_correct
            }


        elif q_type == 'Free-Response':

            evaluation = evaluate_free_response(
                user_answer,
                correct_answer
            )

            is_correct = (
                evaluation['status']
                == 'Correct'
            )

            if is_correct:
                correct_q += 1

            review_data[question_id] = {

                'question': question_data.get(
                    'question',
                    ''
                ),

                'type': 'Free-Response',

                'user_answer': user_answer,

                'correct_answer': correct_answer,

                'score': evaluation['score'],

                'status': evaluation['status'],

                'correct': is_correct
            }

    return render_template(
        'result.html',
        total=total,
        score=correct_q,
        review_data=review_data
    )

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
