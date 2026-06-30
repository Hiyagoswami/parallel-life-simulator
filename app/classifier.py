"""
classifier.py — Merchant keyword categorization

Categorizes bank transaction descriptions into 7 spending categories
using keyword matching. ~89% accuracy on Chase/Amex CSV exports,
validated against 100 manually labeled transactions (see validation.csv
and validate.py for the actual measurement).

Reducibility weights are a heuristic mapping calibrated by eye against
BLS Consumer Expenditure Survey 2023 discretionary-vs-fixed spending
categories (https://www.bls.gov/cex/) — NOT a fitted regression. They
reflect a reasoned estimate of how easily each category can be reduced,
not a statistically derived coefficient. Stated explicitly here to avoid
overclaiming precision the project doesn't have.

Known tradeoff: "mobil" was removed from Transportation keywords because
it falsely matched "T-MOBILE" (substring collision). This means standalone
Mobil gas stations now fall through to "Other" instead of Transportation —
a real and acknowledged false-negative traded for fixing a false-positive
that was more common in the validation set. See validate.py for the
measured impact.
"""

CATEGORY_KEYWORDS = {
    "Food & Dining": [
        "starbucks","chipotle","mcdonald","uber eats","doordash","grubhub","panera",
        "whole foods","trader joe","subway","chick-fil","panda express","dunkin",
        "domino","shake shack","sweetgreen","cheesecake","olive garden","pizza",
        "restaurant","cafe","coffee","dining","food","taco","burger","sushi",
        "postmates","instacart","deli","bakery","smoothie","wendy","taco bell",
    ],
    "Shopping": [
        "amazon","target","walmart","zara","h&m","nike","apple store","best buy",
        "ikea","nordstrom","tj maxx","costco","etsy","macy","gap","shein",
        "uniqlo","forever 21","ross","marshalls","ebay","shopify","wayfair",
        "home depot","lowe","bed bath","dollar","five below","adidas",
    ],
    "Subscriptions": [
        "netflix","spotify","hulu","disney","apple music","youtube premium",
        "amazon prime","hbo","planet fitness","adobe","microsoft 365","icloud",
        "dropbox","slack","zoom","linkedin","duolingo","calm","headspace",
        "paramount","peacock","crunchyroll","nytimes","wsj","audible",
    ],
    "Transportation": [
        "uber","lyft","shell","citgo","chevron","exxon","mobil gas","marathon",
        "metra","cta","parking","ez pass","ezpass","jiffy lube","enterprise",
        "hertz","avis","amtrak","greyhound","gas station","toll","transit",
    ],
    "Health & Wellness": [
        "cvs","walgreens","equinox","classpass","doctor","dentist","optum",
        "goodrx","pharmacy","clinic","hospital","gym","yoga","pilates",
        "vitamin","supplement","therapy","urgent care","labcorp","quest",
    ],
    "Entertainment": [
        "amc","ticketmaster","stubhub","steam","playstation","xbox","nintendo",
        "bowling","escape room","museum","comedy","concert","event","movie",
        "theater","sport","golf","arcade","airbnb","vrbo","hotel",
    ],
    "Utilities": [
        "comed","peoples gas","at&t","verizon","comcast","xfinity","t-mobile",
        "sprint","electric","gas company","water","internet","cable","phone",
        "insurance","geico","state farm","allstate","progressive",
    ],
    # FIX #5 — Fixed/Excluded category: rent, payroll, transfers should never
    # be treated as discretionary spend or included in "Total Spent"
    "Fixed/Excluded": [
        "rent","mortgage","payroll","direct deposit","ach credit","ach deposit",
        "transfer to","transfer from","internal transfer","credit card payment",
        "loan payment","student loan","autopay","property tax","hoa fee",
        "venmo","zelle","paypal transfer","savings transfer","401k","ira contribution",
    ],
}

# Reducibility weights — heuristic, BLS CEX-informed, NOT a regression coefficient.
# Fixed/Excluded gets 0.0: these are never treated as a savings opportunity.
REDUCIBILITY = {
    "Subscriptions":    0.90,
    "Shopping":         0.60,
    "Entertainment":    0.55,
    "Food & Dining":    0.35,
    "Transportation":   0.25,
    "Health & Wellness":0.20,
    "Utilities":        0.10,
    "Fixed/Excluded":   0.00,  # never a savings lever
    "Other":            0.30,
}


def infer_category(description: str) -> str:
    """Classify a merchant description into one of 8 categories (7 + Fixed/Excluded)."""
    desc = str(description).lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(k in desc for k in keywords):
            return cat
    return "Other"


def reducibility_label(score: float) -> str:
    if score == 0:   return "Fixed — not a savings lever"
    if score >= 0.8: return "Very easy to cut"
    if score >= 0.5: return "Fairly cuttable"
    if score >= 0.3: return "Some flexibility"
    return "Mostly fixed"


def action_label(category: str, reduction_pct: float) -> str:
    pct = int(reduction_pct * 100)
    labels = {
        "Subscriptions":    "Cancel subscriptions",
        "Shopping":         f"Cut shopping {pct}%",
        "Entertainment":    f"Cut entertainment {pct}%",
        "Food & Dining":    f"Cut food & dining {pct}%",
        "Transportation":   f"Reduce transport {pct}%",
        "Health & Wellness":f"Trim wellness {pct}%",
        "Utilities":        f"Reduce utilities {pct}%",
        "Other":            f"Reduce other {pct}%",
    }
    return labels.get(category, f"Reduce {category} {pct}%")
