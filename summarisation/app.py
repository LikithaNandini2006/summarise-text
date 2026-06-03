from flask import Flask, render_template, request
from summarizer import summarize_text_with_accuracy

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    summary = ""
    accuracy = None
    original_text = ""
    error = ""

    if request.method == "POST":
        original_text = request.form.get("text", "")

        if not original_text.strip():
            error = "Please enter some text to summarize."
        else:
            summary, accuracy = summarize_text_with_accuracy(original_text)

    return render_template(
        "index.html",
        summary=summary,
        accuracy=accuracy,
        original_text=original_text,
        error=error,
    )

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
