# SMS Spam Classifier

Reads an SMS message and says **spam** or **ham** (a normal message),
with how confident it is:

```
$ python src/predict.py "WINNER!! Claim your free prize now, text WIN to 80086"
SPAM (95.3% confident)
```

## How it works

The model is trained on the
[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
dataset: 5,574 labelled messages, about 13% spam. After dropping 403
duplicate messages, it splits the rest 80% for training and 20% for
testing.

The model is a scikit-learn `Pipeline` with two steps:

1. **TF-IDF** turns each message into numbers, one score per word for
   how important that word is to the message.
2. **Logistic Regression** turns those numbers into ham or spam.

Keeping both steps in one pipeline means training and prediction always
run the same steps.

## Project layout

```
src/
  download_data.py   fetch the dataset into data/
  data.py            load, de-duplicate and split the data
  model.py           build the TF-IDF + classifier pipeline
  train.py           compare models, save the final one to models/
  evaluate.py        confusion matrices, precision/recall/F1, mistakes
  predict.py         classify your own messages
tests/               pytest tests for each module
notebooks/           data exploration
```

## Setup

You need Python 3.13. Run these commands from the project folder:

```
python3 -m venv .venv            # create a virtual environment
source .venv/bin/activate        # turn it on (macOS/Linux)
pip install -r requirements.txt  # install the libraries
python src/download_data.py      # download the dataset into data/
```

## Train

```
python src/train.py
```

This prints a comparison of the models and saves the final one to
`models/spam_pipeline.joblib`.

To see the detailed scores, the 5-fold cross-validation and every
misclassified test message, run:

```
python src/evaluate.py
```

## Predict

Classify one message:

```
python src/predict.py "Are we still on for lunch tomorrow?"
HAM (94.7% confident)
```

Run it without a message to type messages one after another. Type `quit`
to stop:

```
python src/predict.py
```

## Run the tests

```
pytest
```

## Results

Scored on the test set: 1,035 messages the models never saw during
training, 131 of them spam. Precision, recall and F1 are for the spam class.

| Model                 | Accuracy | Precision | Recall | F1    | 5-fold CV F1 |
|-----------------------|---------:|----------:|-------:|------:|-------------:|
| Always ham (baseline) |    87.3% |      0.0% |   0.0% |  0.0% |         0.0% |
| Naive Bayes           |    95.3% |    100.0% |  62.6% | 77.0% |        75.0% |
| Logistic Regression   |    96.4% |     96.1% |  74.8% | 84.1% |        84.1% |

- **Precision**: of the messages flagged as spam, how many really were spam.
- **Recall**: of the real spam, how much was caught.
- **F1**: one number that combines precision and recall.
- **5-fold CV F1**: average F1 from cross-validation on the training set.

### Why accuracy alone misleads

Only about 13% of messages are spam. A "model" that always answers ham is
87.3% accurate while catching zero spam, so accuracy hides the one thing we
care about. Recall and precision on the spam class show it.

### Final model: Logistic Regression

- It catches more spam: 98 of 131, against 82 of 131 for Naive Bayes.
- It has the best F1 on the test set (84.1% against 77.0%).
- Cross-validation agrees: it beat Naive Bayes on all 5 folds, so the
  result is not a lucky split.

The trade-off: it flagged 4 real messages as spam, where Naive Bayes
flagged none. Blocking a real message is worse than letting one spam
through, so moving the decision threshold is a planned improvement.

### What the mistakes look like

- Missed spam (33) is mostly spam without the usual trigger words: adult
  chat lines, fake delivery or "missed call" notices, quiz questions, and
  very short texts like `ringtoneking 84484`.
- Wrongly flagged ham (4) used words that are common in spam, such as
  `free` and `call` ("Are you free now?can i call now?").
- A few messages labelled spam in the dataset read like ordinary texts,
  so some mistakes come from the labels rather than the model.

## Testing my own messages

| Message                                                 | Model says | Right? |
|---------------------------------------------------------|------------|--------|
| WINNER!! Claim your free prize now, text WIN to 80086   | SPAM 95.3% | Yes    |
| Hey how are you                                         | HAM 98.2%  | Yes    |
| Its free, you may please come                           | HAM 85.8%  | Yes    |
| free stuffs available for your                          | HAM 62.6%  | No     |

Why `free stuffs available for your` slipped through:

- Every message starts at a score of -2.53, leaning ham, because 87% of
  the training messages were ham.
- `free` (+3.26), `your` (+2.35) and `for` (+1.71) all push toward spam,
  but in a short message they add up to only +2.01. The final score of
  -0.52 is still below 0, so the answer is ham.
- `stuffs` never appears in the training data, so it counts for nothing.

Takeaways: one spam word is not enough, short messages carry little
evidence, and the low confidence (62.6%) was a sign the model was unsure.
Lowering the spam threshold or adding word pairs such as `free stuff`
could catch messages like this.
