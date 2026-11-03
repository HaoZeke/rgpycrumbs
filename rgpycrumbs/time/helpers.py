import datetime


def one_day_tdelta(etime: str, stime: str, format_time: str = "%H:%M:%S"):
    """Compute the timedelta between two times within the same day.

    .. versionadded:: 0.0.2
    .. versionchanged:: 1.14.0
       Works on Python 3.10, which has no ``datetime.UTC``.
    """
    # Chain the timezone injection directly to creation
    start_datetime = datetime.datetime.strptime(stime, format_time).replace(
        tzinfo=datetime.timezone.utc
    )
    end_datetime = datetime.datetime.strptime(etime, format_time).replace(
        tzinfo=datetime.timezone.utc
    )

    delta = end_datetime - start_datetime

    # Handle the midnight wrap-around
    if delta.total_seconds() < 0:
        delta += datetime.timedelta(days=1)

    return delta
