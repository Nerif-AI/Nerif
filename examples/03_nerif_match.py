from nerif.core import nerif_match_string

selections = ["iPhone 5", "iPhone 6", "iPhone 12"]

best_choice = nerif_match_string(selections=selections, text="Which iPhone is the most powerful one?")

print(best_choice)

match best_choice:
    case 0:
        print("iPhone 5")
    case 1:
        print("iPhone 6")
    case 2:
        print("iPhone 12")
    case _:
        print("No match")

print("end")
