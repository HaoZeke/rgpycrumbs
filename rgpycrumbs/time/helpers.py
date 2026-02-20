import datetime


def one_day_tdelta(etime: str, stime: str, format_time: str = "%H:%M:%S"):
    # Chain the timezone injection directly to creation
    start_datetime = datetime.datetime.strptime(stime, format_time).replace(
        tzinfo=datetime.UTC
    )
    end_datetime = datetime.datetime.strptime(etime, format_time).replace(
        tzinfo=datetime.UTC
    )

    delta = end_datetime - start_datetime

    # Handle the midnight wrap-around
    if delta.total_seconds() < 0:
        delta += datetime.timedelta(days=1)

    return delta
