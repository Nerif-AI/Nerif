from nerif.utils import (
    FormatVerifierFloat,
    FormatVerifierHumanReadableList,
    FormatVerifierInt,
    FormatVerifierListInt,
    NerifFormat,
)

dummy_response1 = "1"
dummy_response2 = "The result is: 1"
dummy_response3 = "1.0"
dummy_response4 = "The result is: 1.0"
dummy_response5 = "[1,2,3]"
dummy_response6 = "The result is: [1,2,3]"

formatter = NerifFormat()

result1 = formatter.try_convert(dummy_response1, FormatVerifierInt)
result2 = formatter.try_convert(dummy_response2, FormatVerifierInt)
result3 = formatter.try_convert(dummy_response3, FormatVerifierFloat)
result4 = formatter.try_convert(dummy_response4, FormatVerifierFloat)
result5 = formatter.try_convert(dummy_response5, FormatVerifierListInt)
result6 = formatter.try_convert(dummy_response6, FormatVerifierListInt)
print(dummy_response1, "->", result1)
print(dummy_response2, "->", result2)
print(dummy_response3, "->", result3)
print(dummy_response4, "->", result4)
print(dummy_response5, "->", result5)
print(dummy_response6, "->", result6)

failed_response1 = "There is no available result"
try:
    result7 = formatter.try_convert(failed_response1, FormatVerifierInt)
except ValueError:
    print('We cannot convert the response "{}" to int'.format(failed_response1))

failed_response2 = "The result is: 114514"
try:
    result8 = formatter.try_convert(failed_response2, FormatVerifierFloat)
except ValueError:
    print('We cannot convert the response "{}" to float'.format(failed_response2))

human_readable_list = """
Here are some fluits:
    1. Apple
    2. Banana
    3. Cherry
    4. Durian
5. Elderberry
6.     Fig
"""
result_list = formatter.try_convert(human_readable_list, FormatVerifierHumanReadableList)
print(result_list)


#################### Be aware the following condition may incur an error without exception ####################
warning_response = "The result is: 114514.1919810"
result8 = formatter.try_convert(warning_response, FormatVerifierInt)
print(warning_response, "->", result8)
