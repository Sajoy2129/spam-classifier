"""Classify your own messages as SPAM or HAM with the saved model.

One message:    python src/predict.py "You won a free prize!"
Interactive:    python src/predict.py   (type 'quit' to stop)
"""

import sys
from pathlib import Path

import joblib

from train import MODEL_FILE

QUIT_WORDS = {"quit", "exit", "q"}


def load_model(path=MODEL_FILE):
    """Load the trained pipeline saved by train.py."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run `python src/train.py` first."
        )
    return joblib.load(path)


def classify(pipeline, message):
    """Return the predicted label and how confident the model is in it.

    predict_proba gives one probability per label, e.g. [0.07, 0.93]
    for [ham, spam]. The confidence is the probability of the winner.
    """
    probabilities = pipeline.predict_proba([message])[0]
    best = probabilities.argmax()  # position of the highest probability
    return pipeline.classes_[best], probabilities[best]


def format_result(label, confidence):
    """Turn a label and confidence into text like 'SPAM (93.2% confident)'."""
    return f"{label.upper()} ({confidence:.1%} confident)"


def interactive_loop(pipeline):
    """Keep asking for messages until the user types quit."""
    print("Type a message to classify it, or 'quit' to stop.")
    while True:
        try:
            message = input("\nMessage: ").strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl+D or Ctrl+C
            print()
            break

        if message.lower() in QUIT_WORDS:
            break
        if not message:
            continue  # ignore empty lines

        print(format_result(*classify(pipeline, message)))


def main(args):
    """Classify the message in args, or start the interactive loop."""
    try:
        pipeline = load_model()
    except FileNotFoundError as error:
        print(error)
        sys.exit(1)

    if args:
        # Join the words, so the message works even without quotes.
        print(format_result(*classify(pipeline, " ".join(args))))
    else:
        interactive_loop(pipeline)


if __name__ == "__main__":
    main(sys.argv[1:])  # sys.argv[0] is the script name, so skip it
