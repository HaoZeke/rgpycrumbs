import re

_NUM = re.compile(
    r"""
    [-+]?                    # Optional sign
    (?:\d*\.\d+|\d+\.?)      # Integer or decimal part
    (?:[eE][-+]?\d+)?        # Optional scientific notation part
""",
    re.X,
)
