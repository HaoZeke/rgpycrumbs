``rgpycrumbs.time.helpers.one_day_tdelta`` works on Python 3.10 again.
It used ``datetime.UTC``, which arrived in Python 3.11, and raised ``AttributeError`` on 3.10.
