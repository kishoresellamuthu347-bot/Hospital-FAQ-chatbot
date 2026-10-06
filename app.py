from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from faq_data import faq_data


app = Flask(__name__)


# --------------------------------
# Create FAQ question list
# --------------------------------

questions = []

for item in faq_data:
    questions.extend(item["questions"])


# --------------------------------
# Convert FAQ questions into TF-IDF
# --------------------------------

vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(questions)


# --------------------------------
# Keyword-based intent detection
# --------------------------------

def detect_intent(user_question):

    text = user_question.lower()


    # Appointment booking

    if "appointment" in text:

        if any(word in text for word in [
            "book",
            "get",
            "schedule",
            "need",
            "want",
            "make"
        ]):

            return "book_appointment"


    # Cancel appointment

    if "cancel" in text and "appointment" in text:

        return "cancel_appointment"


    # Doctor availability

    if (
        "doctor" in text
        and any(word in text for word in [
            "when",
            "time",
            "available",
            "arrive",
            "come"
        ])
    ):

        return "doctor_availability"


    # Hospital location

    if (
        "hospital" in text
        and any(word in text for word in [
            "where",
            "location",
            "located",
            "address"
        ])
    ):

        return "hospital_location"


    # Pharmacy

    if any(word in text for word in [
        "pharmacy",
        "medicine",
        "medical shop"
    ]):

        return "pharmacy_location"


    # Laboratory

    if any(word in text for word in [
        "laboratory",
        "lab"
    ]):

        return "laboratory_location"


    # Visiting hours

    if (
        "visit" in text
        or "visiting" in text
    ):

        return "visiting_hours"


    # Patient registration

    if (
        "register" in text
        or "registration" in text
        or "new patient" in text
    ):

        return "patient_registration"


    # Medical reports

    if (
        "report" in text
        or "reports" in text
    ):

        return "medical_reports"


    # Payment

    if (
        "payment" in text
        or "pay" in text
        or "card" in text
        or "cash" in text
    ):

        return "payment"


    # Parking

    if "parking" in text:

        return "parking"


    # Emergency

    if (
        "emergency" in text
        or "urgent" in text
    ):

        return "emergency"


    return None


# --------------------------------
# Find answer using intent
# --------------------------------

def get_answer_by_intent(intent):

    for item in faq_data:

        if item["intent"] == intent:

            return item["answer"]

    return None


# --------------------------------
# Home page
# --------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# --------------------------------
# Chat API
# --------------------------------

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    user_question = data.get(
        "question",
        ""
    ).strip()


    if not user_question:

        return jsonify({
            "answer": "Please enter a question."
        })


    # --------------------------------
    # First try keyword-based intent
    # --------------------------------

    detected_intent = detect_intent(
        user_question
    )


    if detected_intent:

        answer = get_answer_by_intent(
            detected_intent
        )

        if answer:

            return jsonify({
                "answer": answer
            })


    # --------------------------------
    # If intent not found,
    # use TF-IDF similarity
    # --------------------------------

    user_vector = vectorizer.transform(
        [user_question]
    )


    similarity_scores = cosine_similarity(
        user_vector,
        faq_vectors
    )


    best_match_index = similarity_scores.argmax()

    best_score = similarity_scores[
        0
    ][
        best_match_index
    ]


    # --------------------------------
    # Similarity threshold
    # --------------------------------

    if best_score < 0.4:

        return jsonify({

            "answer":
            "Sorry, I could not find a suitable answer to your question."

        })


    # --------------------------------
    # Find corresponding FAQ answer
    # --------------------------------

    count = 0

    answer = None


    for item in faq_data:

        for question in item["questions"]:

            if count == best_match_index:

                answer = item["answer"]

                break

            count += 1


        if answer is not None:

            break


    return jsonify({

        "answer": answer

    })


# --------------------------------
# Run application
# --------------------------------

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5000
    )