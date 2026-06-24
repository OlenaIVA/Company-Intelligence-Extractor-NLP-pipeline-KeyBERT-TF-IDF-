"""
Company Intelligence Extractor

What it does:
  1. Fetch and cache company homepage HTML
  2. Extract clean visible text
  3. Extract keywords with KeyBERT or TF-IDF fallback
  4. Extract named entities with spaCy, if installed
  5. Classify companies into sectors with TF-IDF + Logistic Regression
  6. Evaluate model with Stratified CV and train/test split
  7. Save results, probabilities, metrics, model and charts

"""
# Library imports
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline


# KeyBERT 
# KeyBERT is used as the primary keyword extraction method.
# TF-IDF is provided as a lightweight fallback when KeyBERT
# dependencies are unavailable.
try:
    from keybert import KeyBERT

    kw_model = KeyBERT()
    USE_KEYBERT = True
except ImportError:
    kw_model = None
    USE_KEYBERT = False
    print("KeyBERT not installed — falling back to TF-IDF keywords.")


# spaCy NER 
try:
    import spacy

    nlp = spacy.load("en_core_web_sm")
    USE_SPACY = True

except (ImportError, OSError):
    nlp = None
    USE_SPACY = False

# Project paths and configuration
# Centralised configuration for directories, file locations,
# model storage and HTTP request settings used throughout the project.
DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "html_cache"
OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")

RAW_PAGES_PATH = DATA_DIR / "raw_pages.csv"
RESULTS_PATH = OUTPUT_DIR / "company_intelligence.csv"
PROBABILITIES_PATH = OUTPUT_DIR / "sector_probabilities.csv"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.csv"
MODEL_PATH = MODEL_DIR / "sector_classifier.joblib"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; CompanyIntelligenceBot/2.0; "
        "+https://example.com/bot-info)"
    )
}

TIMEOUT = 10
DELAY = 1.5
MIN_TEXT_LEN = 100
MAX_TEXT_CHARS = 3000
TEST_SIZE = 0.2
RANDOM_STATE = 42


# Company List 
# Sample collection of UK companies manually labelled by sector.
# These records are used to download website content, extract features,
# and train the company sector classification model.
COMPANIES = [
    # FinTech
    {"name": "Monzo", "url": "https://monzo.com", "sector": "FinTech"},
    {"name": "Revolut", "url": "https://revolut.com", "sector": "FinTech"},
    {"name": "Starling Bank", "url": "https://starlingbank.com", "sector": "FinTech"},
    {"name": "Wise", "url": "https://wise.com", "sector": "FinTech"},
    {"name": "OakNorth", "url": "https://oaknorth.com", "sector": "FinTech"},
    {"name": "Zopa", "url": "https://www.zopa.com", "sector": "FinTech"},
    {"name": "Funding Circle", "url": "https://www.fundingcircle.com/uk", "sector": "FinTech"},
    {"name": "ClearBank", "url": "https://www.clear.bank", "sector": "FinTech"},
    {"name": "GoCardless", "url": "https://gocardless.com", "sector": "FinTech"},
    {"name": "SumUp", "url": "https://www.sumup.com", "sector": "FinTech"},
    {"name": "Tide", "url": "https://www.tide.co", "sector": "FinTech"},
    {"name": "TrueLayer", "url": "https://truelayer.com", "sector": "FinTech"},
    {"name": "Checkout.com", "url": "https://www.checkout.com", "sector": "FinTech"},
    {"name": "Curve", "url": "https://www.curve.com", "sector": "FinTech"},
    {"name": "Moneybox", "url": "https://www.moneyboxapp.com", "sector": "FinTech"},

    # HealthTech 
    {"name": "Elvie", "url": "https://elvie.com", "sector": "HealthTech"},
    {"name": "Huma", "url": "https://huma.com", "sector": "HealthTech"},
    {"name": "Accurx", "url": "https://accurx.com", "sector": "HealthTech"},
    {"name": "BenevolentAI", "url": "https://benevolent.com", "sector": "HealthTech"},
    {"name": "MedShr", "url": "https://medshr.net", "sector": "HealthTech"},
    {"name": "Cera", "url": "https://www.ceracare.co.uk", "sector": "HealthTech"},
    {"name": "Livi", "url": "https://www.livi.co.uk", "sector": "HealthTech"},
    {"name": "Doctor Care Anywhere", "url": "https://doctorcareanywhere.com", "sector": "HealthTech"},
    {"name": "Skin Analytics", "url": "https://skin-analytics.com", "sector": "HealthTech"},
    {"name": "Kheiron Medical", "url": "https://kheironmed.com", "sector": "HealthTech"},
    {"name": "Oxford Nanopore", "url": "https://nanoporetech.com", "sector": "HealthTech"},
    {"name": "Healx", "url": "https://healx.io", "sector": "HealthTech"},
    {"name": "Peppy", "url": "https://peppy.health", "sector": "HealthTech"},
    {"name": "Birdie", "url": "https://www.birdie.care", "sector": "HealthTech"},
    {"name": "Thriva", "url": "https://thriva.co", "sector": "HealthTech"},

    # GreenTech 
    {"name": "Octopus Energy", "url": "https://octopus.energy", "sector": "GreenTech"},
    {"name": "Zapmap", "url": "https://www.zap-map.com", "sector": "GreenTech"},
    {"name": "Ripple Energy", "url": "https://rippleenergy.com", "sector": "GreenTech"},
    {"name": "OVO Energy", "url": "https://www.ovoenergy.com", "sector": "GreenTech"},
    {"name": "Pod Point", "url": "https://pod-point.com", "sector": "GreenTech"},
    {"name": "Zenobe", "url": "https://zenobe.com", "sector": "GreenTech"},
    {"name": "Gridserve", "url": "https://gridserve.com", "sector": "GreenTech"},
    {"name": "Connected Energy", "url": "https://connected-energy.co.uk", "sector": "GreenTech"},
    {"name": "Bulb", "url": "https://bulb.co.uk", "sector": "GreenTech"},
    {"name": "Moixa", "url": "https://www.moixa.com", "sector": "GreenTech"},
    {"name": "Kraken Technologies", "url": "https://kraken.tech", "sector": "GreenTech"},
    {"name": "GeoPura", "url": "https://geopura.com", "sector": "GreenTech"},
    {"name": "Carbon Clean", "url": "https://carbonclean.com", "sector": "GreenTech"},
    {"name": "Dendra Systems", "url": "https://dendra.io", "sector": "GreenTech"},

    # E-commerce 
    {"name": "ASOS", "url": "https://www.asos.com", "sector": "E-commerce"},
    {"name": "Farfetch", "url": "https://www.farfetch.com", "sector": "E-commerce"},
    {"name": "THG", "url": "https://www.thg.com", "sector": "E-commerce"},
    {"name": "Moonpig", "url": "https://www.moonpig.com", "sector": "E-commerce"},
    {"name": "Boohoo", "url": "https://www.boohoo.com", "sector": "E-commerce"},
    {"name": "Ocado", "url": "https://www.ocadogroup.com", "sector": "E-commerce"},
    {"name": "Not On The High Street", "url": "https://www.notonthehighstreet.com", "sector": "E-commerce"},
    {"name": "Secret Sales", "url": "https://www.secretsales.com", "sector": "E-commerce"},
    {"name": "AO.com", "url": "https://ao.com", "sector": "E-commerce"},
    {"name": "Hotel Chocolat", "url": "https://www.hotelchocolat.com", "sector": "E-commerce"},
    {"name": "JD Sports", "url": "https://www.jdsports.co.uk", "sector": "E-commerce"},
    {"name": "Gymshark", "url": "https://www.gymshark.com", "sector": "E-commerce"},
    {"name": "Marks & Spencer", "url": "https://www.marksandspencer.com", "sector": "E-commerce"},
    {"name": "Very", "url": "https://www.very.co.uk", "sector": "E-commerce"},

    # AI / Data 
    {"name": "Google DeepMind", "url": "https://deepmind.google", "sector": "AI/Data"},
    {"name": "Faculty AI", "url": "https://faculty.ai", "sector": "AI/Data"},
    {"name": "Featurespace", "url": "https://featurespace.com", "sector": "AI/Data"},
    {"name": "Graphcore", "url": "https://www.graphcore.ai", "sector": "AI/Data"},
    {"name": "Quantexa", "url": "https://www.quantexa.com", "sector": "AI/Data"},
    {"name": "Synthesia", "url": "https://www.synthesia.io", "sector": "AI/Data"},
    {"name": "Multiverse", "url": "https://www.multiverse.io", "sector": "AI/Data"},
    {"name": "Peak", "url": "https://peak.ai", "sector": "AI/Data"},
    {"name": "Secondmind", "url": "https://www.secondmind.ai", "sector": "AI/Data"},
    {"name": "Signal AI", "url": "https://www.signal-ai.com", "sector": "AI/Data"},
    {"name": "Builder.ai", "url": "https://www.builder.ai", "sector": "AI/Data"},
    {"name": "Tractable", "url": "https://tractable.ai", "sector": "AI/Data"},
    {"name": "Eigen Technologies", "url": "https://eigen.tech", "sector": "AI/Data"},
    {"name": "Luminance", "url": "https://www.luminance.com", "sector": "AI/Data"},

    # Cybersecurity 
    {"name": "Darktrace", "url": "https://darktrace.com", "sector": "Cybersecurity"},
    {"name": "Snyk", "url": "https://snyk.io", "sector": "Cybersecurity"},
    {"name": "Immersive", "url": "https://www.immersivelabs.com", "sector": "Cybersecurity"},
    {"name": "NCC Group", "url": "https://www.nccgroup.com", "sector": "Cybersecurity"},
    {"name": "Bridewell", "url": "https://www.bridewell.com", "sector": "Cybersecurity"},
    {"name": "OutThink", "url": "https://outthink.com", "sector": "Cybersecurity"},
    {"name": "Six Degrees", "url": "https://www.6dg.co.uk", "sector": "Cybersecurity"},
    {"name": "Secureworks", "url": "https://www.secureworks.com", "sector": "Cybersecurity"},
    {"name": "Claranet", "url": "https://www.claranet.com", "sector": "Cybersecurity"},
    {"name": "CybSafe", "url": "https://www.cybsafe.com", "sector": "Cybersecurity"},
]


# Utilities 
def ensure_directories() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)


def is_usable_text(text: Any) -> bool:
    return isinstance(text, str) and bool(text.strip())


def is_long_enough(text: Any) -> bool:
    return is_usable_text(text) and len(text.strip()) >= MIN_TEXT_LEN


def url_to_cache_path(url: str) -> Path:
    url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{url_hash}.html"


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


# Scraper 
# Common class/id substrings used by cookie-consent banners, popups, and
# similar chrome that isn't really "page content" but isn't caught by
# semantic tags like <nav>/<footer>/<header> either.
NOISE_SELECTOR_HINTS = (
    "cookie",
    "consent",
    "gdpr",
    "popup",
    "modal",
    "banner",
    "newsletter-signup",
)

# Extra stop-words to drop from TF-IDF / keyword output — site chrome and
# UI copy that survives clean_text() but isn't meaningful business content.
EXTRA_STOP_WORDS = {
    "cookie", "cookies", "consent", "accept", "decline", "preferences",
    "optional", "necessary", "browser", "javascript", "enable",
    "skip", "menu", "search", "close", "video", "tag",
}

# sklearn's built-in "english" stop-word list, extended with the noise
# words above. Used everywhere instead of the bare "english" string so
# site-chrome leftovers (cookie banners, nav fragments) get filtered
# consistently across keyword extraction and the classifier.
COMBINED_STOP_WORDS = sorted(set(ENGLISH_STOP_WORDS) | EXTRA_STOP_WORDS)


def _looks_like_noise(tag) -> bool:
     """
    Identify HTML elements that are likely to contain website noise
    rather than meaningful business content.

    The function checks whether a tag's class names or ID contain
    common indicators of cookie banners, consent dialogs, popups,
    newsletters or other user-interface elements.

    Returns:
        True if the element appears to be website noise,
        otherwise False.
    """
    css_class = tag.get("class", []) or []
    tag_id = tag.get("id", "") or ""
    attrs = (" ".join(css_class) + " " + tag_id).lower()
    return any(hint in attrs for hint in NOISE_SELECTOR_HINTS)


def clean_text(html: str) -> str:
    """Extract visible text from raw HTML."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
        tag.decompose()

    # Cookie banners / consent popups / modals are usually plain <div>s, so
    # the tag-name filter above misses them — catch them by class/id instead.
    # find_all() snapshots the list up front, so decomposing a parent here
    # can leave its still-listed children detached (decomposed) by the time
    # we reach them — skip anything that's no longer attached to the tree.
    for tag in soup.find_all(["div", "section", "aside"]):
        if tag.parent is None:
            continue
        if _looks_like_noise(tag):
            tag.decompose()

    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_TEXT_CHARS]


def fetch_html(session: requests.Session, url: str, use_cache: bool = True) -> dict[str, Any]:
    """Fetch HTML with optional file-based cache.

    Note: when served from cache, http_status is None (we don't re-request
    the page, so we have no fresh status code to report).
    """
    cache_path = url_to_cache_path(url)

    if use_cache and cache_path.exists():
        html = cache_path.read_text(encoding="utf-8", errors="ignore")
        return {
            "html": html,
            "http_status": None,
            "error": "",
            "from_cache": True,
        }

    try:
        response = session.get(url, timeout=TIMEOUT)
        response.raise_for_status()

        html = response.text
        cache_path.write_text(html, encoding="utf-8")

        return {
            "html": html,
            "http_status": response.status_code,
            "error": "",
            "from_cache": False,
        }

    except requests.RequestException as exc:
        print(f"  ⚠️ {url} — {exc}")
        return {
            "html": "",
            "http_status": None,
            "error": str(exc),
            "from_cache": False,
        }


def fetch_homepage_text(
    session: requests.Session,
    url: str,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Fetch homepage HTML, clean it and return text plus status fields."""
    html_result = fetch_html(session, url, use_cache=use_cache)
    html = html_result.pop("html")

    if not html:
        return {**html_result, "text": "", "char_count": 0, "fetch_status": "failed"}

    text = clean_text(html)
    return {
        **html_result,
        "text": text,
        "char_count": len(text),
        "fetch_status": "ok" if is_long_enough(text) else "too_short",
    }


def scrape_all(companies: list[dict[str, str]], use_cache: bool = True) -> pd.DataFrame:
    ensure_directories()
    session = make_session()

    records = []
    total = len(companies)

    for i, company in enumerate(companies, 1):
        print(f"[{i}/{total}] Fetching {company['name']} …")
        scrape_result = fetch_homepage_text(session, company["url"], use_cache=use_cache)
        records.append({**company, **scrape_result})

        if not scrape_result["from_cache"]:
            time.sleep(DELAY)

    df = pd.DataFrame(records)
    df.to_csv(RAW_PAGES_PATH, index=False)
    print(f"\n✅ Scraped {len(df)} companies → {RAW_PAGES_PATH}")
    return df


# Keyword Extraction 
def tfidf_top_words(
    text: str,
    n: int = 5,
    ngram_range: tuple[int, int] = (1, 2),
) -> list[str]:
    """Return actual top-n TF-IDF terms for one text blob.

    Terms that are purely (or end in) a 4-digit year are dropped — these
    are almost always copyright-notice noise ("© 2026 Acme Ltd") rather
    than meaningful business content.
    """
    if not is_usable_text(text):
        return []

    try:
        vectorizer = TfidfVectorizer(stop_words=COMBINED_STOP_WORDS, ngram_range=ngram_range)
        matrix = vectorizer.fit_transform([text])
        scores = matrix.toarray()[0]
        terms = vectorizer.get_feature_names_out()

        ranked_indices = scores.argsort()[::-1]
        results = []
        for i in ranked_indices:
            if scores[i] <= 0:
                break
            term = terms[i]
            if re.fullmatch(r"(.*\s)?(19|20)\d{2}", term):
                continue
            results.append(term)
            if len(results) >= n:
                break

        return results

    except ValueError:
        return []


def extract_keywords(text: str, n: int = 5) -> list[str]:
    """Extract top keywords using KeyBERT or TF-IDF fallback."""
    if not is_usable_text(text):
        return []

    if USE_KEYBERT and kw_model is not None:
        try:
            keywords = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words=COMBINED_STOP_WORDS,
                top_n=n,
            )
            return [keyword for keyword, _ in keywords]
        except Exception as exc:
            print(f"  !!! KeyBERT failed — using TF-IDF fallback: {exc}")

    return tfidf_top_words(text, n=n)


# Named Entity Recognition 
def extract_entities(
    text: str,
    labels: set[str] | None = None,
    n: int = 10,
) -> list[str]:
    """Extract named entities with spaCy, if available."""
    if labels is None:
        labels = {"ORG", "PRODUCT", "GPE", "LOC"}

    if not USE_SPACY or nlp is None or not is_usable_text(text):
        return []

    doc = nlp(text[:MAX_TEXT_CHARS])
    entities = []
    seen = set()

    for entity in doc.ents:
        cleaned = entity.text.strip()
        key = cleaned.lower()

        if entity.label_ in labels and len(cleaned) > 2 and key not in seen:
            entities.append(cleaned)
            seen.add(key)

        if len(entities) >= n:
            break

    return entities


# Sector Classifier 
def get_labelled_data(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["text"].fillna("").apply(is_long_enough)
    return df[mask].copy()


# Hyperparameters tuned for a small dataset (~10-15 rows per class).
# With max_features=5000 the TF-IDF vocabulary was 70x larger than the
# training set, which let LogisticRegression memorize individual companies
# instead of learning sector-level patterns (100% in-sample accuracy vs.
# ~70% on holdout). Cutting the vocabulary and adding stronger L2
# regularisation forces the model toward more general, frequently-occurring
# terms.
TFIDF_MAX_FEATURES = 800
TFIDF_MIN_DF = 2  # ignore terms that appear in only one company's page
CLASSIFIER_C = 0.5


def build_classifier() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words=COMBINED_STOP_WORDS,
                    ngram_range=(1, 2),
                    max_features=TFIDF_MAX_FEATURES,
                    min_df=TFIDF_MIN_DF,
                    sublinear_tf=True,
                ),
            ),
            (
                "lr",
                LogisticRegression(
                    max_iter=1000,
                    C=CLASSIFIER_C,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def validate_training_data(labelled: pd.DataFrame) -> None:
    if labelled.empty:
        raise ValueError("No valid homepage text found. Cannot train classifier.")

    if labelled["sector"].nunique() < 2:
        raise ValueError("Need at least 2 sectors with valid text to train classifier.")


def can_do_stratified_test_split(labelled: pd.DataFrame) -> bool:
    """Train/test split needs at least 2 rows per class."""
    return labelled["sector"].value_counts().min() >= 2


def evaluate_with_cross_validation(clf: Pipeline, labelled: pd.DataFrame) -> dict[str, Any] | None:
    """Stratified K-Fold CV. sklearn internally clones `clf` for each fold,
    so this never mutates or fits the `clf` instance passed in."""
    min_class_count = int(labelled["sector"].value_counts().min())

    if min_class_count < 2:
        print("\n !!! Not enough samples per class for stratified cross-validation — skipping.")
        return None

    n_splits = min(3, min_class_count)

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scores = cross_val_score(
        clf,
        labelled["text"],
        labelled["sector"],
        cv=cv,
        scoring="accuracy",
    )

    print(f"\n Stratified CV accuracy: {scores.mean():.2f} ± {scores.std():.2f} (cv={n_splits})")

    return {
        "cv": n_splits,
        "accuracy_mean": round(float(scores.mean()), 4),
        "accuracy_std": round(float(scores.std()), 4),
        "scores": [round(float(score), 4) for score in scores],
    }


def evaluate_with_train_test_split(labelled: pd.DataFrame) -> dict[str, Any] | None:
    """Evaluate on a holdout test set when there is enough data.

    This trains a separate, throwaway classifier (`holdout_clf`) purely to
    measure generalisation on unseen data — it is never saved or reused.
    The model that actually gets persisted to MODEL_PATH and used for
    predictions is the one fit on the *full* labelled dataset back in
    `train_classifier`, since with ~25 companies we want every example to
    count towards the final model. That means in-sample metrics are
    optimistic; this holdout score is a more honest (if noisy, given the
    small size) estimate of real-world accuracy.
    """
    if not can_do_stratified_test_split(labelled):
        print(" !!! Not enough samples per class for train/test split — skipping.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        labelled["text"],
        labelled["sector"],
        test_size=TEST_SIZE,
        stratify=labelled["sector"],
        random_state=RANDOM_STATE,
    )

    holdout_clf = build_classifier()
    holdout_clf.fit(X_train, y_train)
    predictions = holdout_clf.predict(X_test)

    report_dict = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    print("\n🧪 Holdout test classification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    return {
        "test_size": TEST_SIZE,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "classification_report": report_dict,
    }


def evaluate_in_sample(clf: Pipeline, labelled: pd.DataFrame) -> dict[str, Any]:
    """In-sample diagnostics on the final, fully-trained model.

    These numbers are optimistic (the model has seen every one of these
    rows during fit), so treat them as a sanity check / confusion-matrix
    source rather than a generalisation estimate — use `holdout_test` in
    the saved metrics for that.
    """
    predictions = clf.predict(labelled["text"])

    report_dict = classification_report(
        labelled["sector"],
        predictions,
        output_dict=True,
        zero_division=0,
    )

    cm = pd.DataFrame(
        confusion_matrix(labelled["sector"], predictions, labels=clf.classes_),
        index=clf.classes_,
        columns=clf.classes_,
    )
    cm.to_csv(CONFUSION_MATRIX_PATH)

    print("\n📋 In-sample classification report:")
    print(classification_report(labelled["sector"], predictions, zero_division=0))

    return {
        "classification_report": report_dict,
        "confusion_matrix_path": str(CONFUSION_MATRIX_PATH),
    }


def train_classifier(df: pd.DataFrame) -> Pipeline:
    """Train the sector classifier and save model + evaluation artefacts.

    Three distinct model fits happen here, by design:
      1. `evaluate_with_cross_validation` — sklearn clones `clf` per fold.
      2. `evaluate_with_train_test_split` — a throwaway `holdout_clf` fit
         only on the train split, scored on the held-out test split.
      3. The final `clf.fit(...)` below — fit on the *entire* labelled
         dataset. This is the model that gets saved to MODEL_PATH and used
         by `predict_sectors`.
    With a small dataset, using every row for the final model matters more
    than holding data back — the CV and holdout scores above exist purely
    to give an honest estimate of how that final model is likely to perform.
    """
    labelled = get_labelled_data(df)
    validate_training_data(labelled)

    clf = build_classifier()

    metrics: dict[str, Any] = {
        "n_training_rows": int(len(labelled)),
        "n_classes": int(labelled["sector"].nunique()),
        "class_distribution": labelled["sector"].value_counts().to_dict(),
        "cross_validation": evaluate_with_cross_validation(clf, labelled),
        "holdout_test": evaluate_with_train_test_split(labelled),
        "in_sample": None,
    }

    clf.fit(labelled["text"], labelled["sector"])
    metrics["in_sample"] = evaluate_in_sample(clf, labelled)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    joblib.dump(clf, MODEL_PATH)

    print(f" Model saved → {MODEL_PATH}")
    print(f" Metrics saved → {METRICS_PATH}")
    print(f" Confusion matrix saved → {CONFUSION_MATRIX_PATH}\n")

    return clf


def predict_sectors(clf: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """Predict sector, confidence and class probabilities."""
    df = df.copy()
    mask = df["text"].fillna("").apply(is_long_enough)

    df["predicted_sector"] = "Unknown"
    df["confidence"] = 0.0

    if not mask.any():
        print(" !!! No valid rows available for prediction.")
        return df

    probabilities = clf.predict_proba(df.loc[mask, "text"])
    predicted_indices = probabilities.argmax(axis=1)

    df.loc[mask, "predicted_sector"] = clf.classes_[predicted_indices]
    df.loc[mask, "confidence"] = probabilities.max(axis=1).round(2)

    proba_df = pd.DataFrame(probabilities, columns=[f"proba_{c}" for c in clf.classes_])
    proba_df.insert(0, "name", df.loc[mask, "name"].values)
    proba_df.insert(1, "true_sector", df.loc[mask, "sector"].values)
    proba_df.insert(2, "predicted_sector", df.loc[mask, "predicted_sector"].values)
    proba_df.to_csv(PROBABILITIES_PATH, index=False)

    print(f" Class probabilities saved → {PROBABILITIES_PATH}")
    return df


# Charts 
PALETTE = {
    "FinTech": "#2D6BE4",
    "HealthTech": "#27AE60",
    "GreenTech": "#F39C12",
    "E-commerce": "#8E44AD",
    "AI/Data": "#E74C3C",
    "Unknown": "#BDC3C7",
    "Cybersecurity": "#34495E",
}
KNOWN_SECTORS = [sector for sector in PALETTE if sector != "Unknown"]


def save_chart(fig, filename: str) -> None:
    path = OUTPUT_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f" --- {path}")


def chart_sector_distribution(df: pd.DataFrame) -> None:
    counts = df["predicted_sector"].value_counts()

    if counts.empty:
        print(" !!! No sector distribution data available — skipping chart.")
        return

    colors = [PALETTE.get(sector, "#BDC3C7") for sector in counts.index]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="white")
    ax.bar_label(bars, padding=4, fontsize=9)
    ax.set_xlabel("Number of companies")
    ax.set_title("Predicted Sector Distribution", fontweight="bold", pad=12)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.spines[["top", "right"]].set_visible(False)

    save_chart(fig, "sector_distribution.png")


def chart_confidence(df: pd.DataFrame) -> None:
    plot_df = df[df["confidence"] > 0].sort_values("confidence", ascending=True).tail(20)

    if plot_df.empty:
        print("!!! No confidence data available — skipping chart.")
        return

    colors = [PALETTE.get(sector, "#BDC3C7") for sector in plot_df["predicted_sector"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(plot_df["name"], plot_df["confidence"], color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=8)
    ax.set_xlabel("Model confidence")
    ax.set_title("Classification Confidence per Company", fontweight="bold", pad=12)
    ax.set_xlim(0, 1.15)
    ax.spines[["top", "right"]].set_visible(False)

    save_chart(fig, "confidence_chart.png")


def chart_keywords_by_sector(df: pd.DataFrame) -> None:
    available_sectors = [
        sector
        for sector in KNOWN_SECTORS
        if not df.loc[df["predicted_sector"] == sector].empty
    ]

    if not available_sectors:
        print("!!! No sector keywords available — skipping chart.")
        return

    fig, axes = plt.subplots(
        1,
        len(available_sectors),
        figsize=(4 * len(available_sectors), 4),
        squeeze=False,
    )
    axes = axes.flatten()

    for ax, sector in zip(axes, available_sectors):
        texts = " ".join(df.loc[df["predicted_sector"] == sector, "text"].dropna())
        words = tfidf_top_words(texts, n=8) if texts else []

        if not words:
            ax.set_visible(False)
            continue

        ax.barh(range(len(words)), [1] * len(words), color=PALETTE[sector], alpha=0.8)
        ax.invert_yaxis()
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=7)
        ax.set_title(sector, fontsize=9, fontweight="bold", color=PALETTE[sector])
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        ax.xaxis.set_visible(False)

    fig.suptitle("Top Keywords by Sector", fontweight="bold", y=1.02)
    save_chart(fig, "keywords_by_sector.png")


def make_charts(df: pd.DataFrame) -> None:
    chart_sector_distribution(df)
    chart_confidence(df)
    chart_keywords_by_sector(df)


# Main 
def main() -> None:
    print("=" * 70)
    print("  Company Intelligence Extractor")
    print("=" * 70 + "\n")

    ensure_directories()

    if RAW_PAGES_PATH.exists():
        print("⚡ Cached raw_pages.csv found — skipping scrape.")
        print(f"   Delete {RAW_PAGES_PATH} to rebuild the dataset.\n")
        df = pd.read_csv(RAW_PAGES_PATH)
    else:
        df = scrape_all(COMPANIES, use_cache=True)

    df["text"] = df["text"].fillna("")

    print("🔍 Extracting keywords …")
    df["keywords"] = df["text"].apply(lambda text: extract_keywords(text, n=5))

    print("🏷️ Extracting named entities …")
    df["entities"] = df["text"].apply(lambda text: extract_entities(text, n=10))

    print("🤖 Training sector classifier …")
    clf = train_classifier(df)

    print("🔮 Predicting sectors …")
    df = predict_sectors(clf, df)

    output_columns = [
        "name",
        "url",
        "sector",
        "predicted_sector",
        "confidence",
        "keywords",
        "entities",
        "char_count",
        "fetch_status",
        "http_status",
        "error",
        "from_cache",
    ]

    existing_columns = [column for column in output_columns if column in df.columns]
    out = df[existing_columns]
    out.to_csv(RESULTS_PATH, index=False)

    print(f"💾 Results saved → {RESULTS_PATH}\n")

    display_columns = [
        col for col in ["name", "sector", "predicted_sector", "confidence", "fetch_status"]
        if col in out.columns
    ]
    print(out[display_columns].to_string(index=False))

    print("\n Generating charts …")
    make_charts(df)

    print("\n  Done! Check the data/, outputs/ and models/ folders.")


if __name__ == "__main__":
    main()
