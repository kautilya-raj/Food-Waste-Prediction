"""utils.py — logging setup and input validation."""
import os, logging, logging.handlers
LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

def setup_logging(log_level="INFO"):
    level = getattr(logging, log_level.upper(), logging.INFO)
    fmt   = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(); ch.setLevel(level); ch.setFormatter(fmt)
    fh = logging.handlers.RotatingFileHandler(os.path.join(LOGS_DIR,"app.log"),maxBytes=5*1024*1024,backupCount=3)
    fh.setLevel(level); fh.setFormatter(fmt)
    root = logging.getLogger(); root.setLevel(level); root.handlers.clear()
    root.addHandler(ch); root.addHandler(fh)

def validate_positive_number(value, field_name):
    try:    v = float(value)
    except: raise ValueError(f"'{field_name}' must be a number, got: {value!r}")
    if v < 0: raise ValueError(f"'{field_name}' must be >= 0")
    return v

def validate_choice(value, field_name, choices):
    v = str(value).lower(); c = [x.lower() for x in choices]
    if v not in c: raise ValueError(f"'{field_name}' must be one of {choices}, got: {value!r}")
    return v

FIELD_SPECS = {
    "hostel": [
        ("num_students",       "int",   None,                                      1,   10000),
        ("meal_type",          "str",   ["breakfast","lunch","dinner"],             None, None),
        ("day_of_week",        "str",   ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],None, None),
        ("season",             "str",   ["summer","winter","monsoon","spring"],     None, None),
        ("special_occasion",   "int",   None,                                      0,   1),
        ("leftover_yesterday", "float", None,                                      0,   500),
    ],
    "hotel": [
        ("num_guests",   "int",   None,                                            1,   10000),
        ("meal_type",    "str",   ["breakfast","lunch","dinner","brunch"],          None, None),
        ("day_of_week",  "str",   ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],      None, None),
        ("season",       "str",   ["summer","winter","monsoon","spring"],           None, None),
        ("event_type",   "str",   ["conference","wedding","regular","gala"],        None, None),
        ("buffet_style", "int",   None,                                            0,   1),
        ("avg_rating",   "float", None,                                            1.0, 5.0),
    ],
    "wedding": [
        ("num_guests",                "int",   None,                                          1, 100000),
        ("cuisine_type",              "str",   ["indian","continental","chinese","mixed"],     None, None),
        ("num_dishes",                "int",   None,                                          1, 200),
        ("catering_experience_years", "int",   None,                                          0, 100),
        ("outdoor_event",             "int",   None,                                          0, 1),
        ("season",                    "str",   ["summer","winter","monsoon","spring"],         None, None),
    ],
    "household": [
        ("family_size",        "int",   None,                                          1,  50),
        ("meal_type",          "str",   ["breakfast","lunch","dinner"],                None, None),
        ("day_of_week",        "str",   ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],   None, None),
        ("shopping_frequency", "str",   ["daily","weekly","biweekly","monthly"],       None, None),
        ("income_level",       "str",   ["low","medium","high"],                       None, None),
        ("season",             "str",   ["summer","winter","monsoon","spring"],        None, None),
    ],
}

def validate_input(venue_type, form_data):
    specs = FIELD_SPECS.get(venue_type.lower())
    if specs is None: raise ValueError(f"Unknown venue_type: {venue_type!r}")
    cleaned = {}
    for (field, dtype, choices, lo, hi) in specs:
        raw = form_data.get(field)
        if raw is None or str(raw).strip() == "": raise ValueError(f"Missing required field: {field!r}")
        if dtype == "str":
            cleaned[field] = validate_choice(raw, field, choices)
        elif dtype == "int":
            v = int(validate_positive_number(raw, field))
            if lo is not None and v < lo: raise ValueError(f"'{field}' must be >= {lo}")
            if hi is not None and v > hi: raise ValueError(f"'{field}' must be <= {hi}")
            cleaned[field] = v
        elif dtype == "float":
            v = validate_positive_number(raw, field)
            if lo is not None and v < lo: raise ValueError(f"'{field}' must be >= {lo}")
            if hi is not None and v > hi: raise ValueError(f"'{field}' must be <= {hi}")
            cleaned[field] = v
    return cleaned
