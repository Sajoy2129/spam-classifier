# SMS Spam Classifier — Project Plan

> Work through one phase at a time. Tick a box (`- [x]`) when the step is done.

## Context

Goal: build a program that reads an SMS message and says **spam** or **ham** (ham = a normal, non-spam message), using the SMS Spam Collection dataset from UCI (5,574 labelled messages, about 13% spam).

This is a first end-to-end machine learning (ML) project, so the plan is built for learning: each phase introduces one idea, ends with something that runs, and ends with a small git commit.

## Approach in one paragraph

Explore the data in a Jupyter notebook, then write the real code as small Python files in `src/`. Turn text into numbers with **TF-IDF** (a score for how important each word is to a message), train a **Naive Bayes** model (a simple, fast classifier that works well on text), compare it against **Logistic Regression**, and judge both with precision and recall rather than accuracy alone. Wrap the vectorizer and model in one scikit-learn **Pipeline** so the same steps run in training and prediction. Finish with a command-line script for typing in your own messages.

## Target layout

```
spam-classifier/
├── PLAN.md
├── README.md
├── requirements.txt
├── .gitignore               # ignores .venv/, data/, models/, __pycache__/
├── data/                    # downloaded dataset (not committed)
├── models/                  # saved trained model (not committed)
├── notebooks/
│   └── 01_explore.ipynb
├── src/
│   ├── download_data.py     # fetch + unzip the dataset
│   ├── data.py              # load_data(), split_data()
│   ├── model.py             # build_pipeline()
│   ├── train.py             # train + save the model
│   ├── evaluate.py          # metrics + confusion matrix
│   └── predict.py           # test your own messages
└── tests/
    ├── test_data.py
    └── test_model.py
```

## Checklist

### Phase 0 — Project setup
- [x] Save this plan as `PLAN.md`
- [x] Add `.gitignore` (`.venv/`, `data/`, `models/`, `__pycache__/`, `.ipynb_checkpoints/`)
- [x] Activate the venv and install `pytest`; refresh `requirements.txt` with `pip freeze`
- [x] Create the `src/`, `tests/`, `notebooks/` folders
- [x] First commit: plan, `.gitignore`, `requirements.txt`

### Phase 1 — Get the data
- [x] Write `src/download_data.py` to download and unzip from `https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip` into `data/`
- [x] Write `load_data()` in `src/data.py`: read the `SMSSpamCollection` file (tab-separated, no header row) into a pandas DataFrame with columns `label` and `message`
- [x] Sanity check: 5,574 rows, only two labels (`ham`, `spam`), no empty messages
- [x] Test in `tests/test_data.py`: columns and labels are what we expect
- [x] Commit

### Phase 2 — Explore the data (`notebooks/01_explore.ipynb`)
- [x] Read 10 ham and 10 spam messages by eye
- [x] Count each class and plot it — expect roughly 87% ham / 13% spam (this is called **class imbalance**)
- [x] Compare message length for spam vs ham (histogram)
- [x] List the most common words in each class
- [x] Check for duplicate messages and decide whether to drop them — decision: drop (403 duplicate rows)
- [x] Write 3–4 bullet-point findings at the bottom of the notebook
- [x] Commit

### Phase 3 — Turn text into numbers
- [x] Drop the 403 duplicate messages before splitting (decided in Phase 2)
- [x] Write `split_data()` in `src/data.py`: 80% train / 20% test, **stratified** (both parts keep the same spam ratio), fixed `random_state` so results repeat
- [x] Try `CountVectorizer` on 3 toy sentences to see the word-count table it makes (bag of words)
- [x] Switch to `TfidfVectorizer` and see how the numbers change
- [x] Learn the key rule: fit the vectorizer on training data only — fitting on test data is **data leakage** (the model peeks at the exam)
- [x] Commit

### Phase 4 — Train a model
- [x] Write `build_pipeline()` in `src/model.py`: `TfidfVectorizer` → `MultinomialNB` in one scikit-learn `Pipeline`
- [x] Write `src/train.py`: load, drop duplicates, split, fit each model, print the comparison
- [x] Add a baseline for comparison: a "model" that always says ham (about 87% accurate while catching zero spam)
- [x] Train Logistic Regression as a second model to compare
- [x] Test in `tests/test_model.py`: pipeline fits on a tiny made-up dataset and predicts a valid label
- [x] Commit

### Phase 5 — Evaluate properly (`src/evaluate.py`)
- [x] Score on the test set only — data the model never saw
- [x] Print the **confusion matrix** (table of right and wrong guesses per class)
- [x] Report **precision** (of messages flagged spam, how many really were), **recall** (of real spam, how much we caught) and **F1** (one number combining both)
- [x] Explain why accuracy alone misleads here, using the always-ham baseline
- [x] Run 5-fold **cross-validation** on the training set (train and score 5 times on different slices) to check the result is not a lucky split
- [x] Read the misclassified messages and look for patterns
- [x] Pick the final model, save it to `models/spam_pipeline.joblib`, and write the numbers and the reason in `README.md`
- [x] Commit

### Phase 6 — Test my own messages (`src/predict.py`)
- [x] Load the saved pipeline with joblib
- [x] One-shot mode: `python src/predict.py "You won a free prize!"` prints the label and spam probability
- [x] Interactive mode: no argument starts a loop that keeps asking for messages until you type `quit`
- [x] Friendly error if the model file does not exist yet ("run train.py first")
- [ ] Try 10 of your own messages, including tricky ones, and note where it fails
- [ ] Commit

### Phase 7 — Wrap up
- [ ] Finish `README.md`: what it does, how to set up, how to train, how to predict, results
- [ ] Run `/code-review` on the project and fix what it finds
- [ ] Final commit

### Stretch ideas (optional, later)
- [ ] Tune settings with `GridSearchCV` (word pairs via `ngram_range`, `min_df`, NB `alpha`)
- [ ] Move the decision threshold to trade recall for precision (blocking a real message is worse than letting one spam through)
- [ ] Small web UI with Streamlit

## Verification

How we know it works end to end:

1. `python src/download_data.py` — `data/SMSSpamCollection` exists
2. `python src/train.py` — prints training summary, creates `models/spam_pipeline.joblib`
3. `python src/evaluate.py` — prints confusion matrix and precision/recall/F1; the model clearly beats the always-ham baseline on spam recall
4. `python src/predict.py "WINNER!! Claim your free prize now, text WIN to 80086"` — spam
5. `python src/predict.py "Are we still on for lunch tomorrow?"` — ham
6. `pytest` — all tests pass
