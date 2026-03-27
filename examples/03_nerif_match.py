from nerif.core import nerif_match_string

selections = ["iPhone 5", "iPhone 6", "iPhone 12"]

result = nerif_match_string(selections=selections, text="Which iPhone is the most powerful one?")

print(result)

if result == 0:
    print("iPhone 5")
elif result == 1:
    print("iPhone 6")
elif result == 2:
    print("iPhone 12")
else:
    print("No match")

print("end")
