import datetime
import pprint
import random
import string

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


def generate_random_string(length=20):
    # string.ascii_letters는 대소문자 알파벳을 모두 포함하며, string.digits는 숫자 0-9를 포함한다.
    characters = string.ascii_letters + string.digits
    # random.choices 함수를 사용하여 주어진 문자열에서 임의의 문자를 선택하고, 이를 지정된 길이만큼 반복한다.
    random_string = ''.join(random.choices(characters, k=length))
    return random_string


# 함수 호출 및 결과 출력
random_str = generate_random_string()
print(random_str)
