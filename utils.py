import datetime
import pprint

import pytz


def show(data):
    pp = pprint.PrettyPrinter(indent=4)
    print("+==================================================================================+")
    pp.pprint(data)
    print()
    # pprint.pprint(data)


def show_in_chain(inputs):
    show(inputs)
    return inputs


def get_current_time_korea_formatted():
    korea_timezone = pytz.timezone('Asia/Seoul')
    current_time_korea = datetime.now(korea_timezone)
    formatted_time = current_time_korea.strftime('%Y-%m-%d-%H:%M:%S')
    return formatted_time
