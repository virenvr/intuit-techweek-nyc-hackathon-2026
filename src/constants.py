"""Fixed loan product terms from the hackathon dataset guide."""

APR = 0.35
TERM_DAYS = 60
ORIGINATION_FEE_RATE = 0.03
DEFAULT_WINDOW_DAYS = 90

DATA_DIR = "dataset/dataset-compressed"
COHORT_DEFINITIONS_PATH = "dataset/cohort_week_definitions.csv"
B_TEMPLATE_PATH = "dataset/submission_B_template.csv"
DATA_DICTIONARY_PATH = "dataset/data_dictionary.csv"
INTERVENTION_QUERIES_PATH = "dataset/intervention_queries.csv"

N_COHORT_WEEKS = 13
N_LOAN_AGE_WEEKS = 13

# Deliverable B v1 failsafes (tune later).
MIN_APPROVED_COHORT_SIZE = 5  # below this, blend with historical KM
MIN_INTERVAL_HALF_WIDTH = 0.02
INTERVAL_Z_SCORE = 1.645  # ~90% normal approximation for binomial CI
DEFAULT_SUBMISSION_A = "submission/submission_A_decisions.csv"

# Deliverable C v1 failsafes (tune later).
NON_INTERVENABLE_BLEND = 0.25  # blend toward observational PD when do() ill-defined
NON_INTERVENABLE_INTERVAL_MULTIPLIER = 1.5
MIN_CF_INTERVAL_HALF_WIDTH = 0.03
