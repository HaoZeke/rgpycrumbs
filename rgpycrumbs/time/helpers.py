import datetime


def one_day_tdelta(etime: str, stime: str, format_time: str = "%H:%M:%S"):
    start_datetime = datetime.datetime.strptime(stime, format_time)
    end_datetime = datetime.datetime.strptime(etime, format_time)
    # XXX: Assumes but cannot check with the default format string
    # that the wrap around is one day
    if end_datetime < start_datetime:
        end_datetime += datetime.timedelta(days=1)
    time_difference = end_datetime - start_datetime
    return time_difference.total_seconds()
