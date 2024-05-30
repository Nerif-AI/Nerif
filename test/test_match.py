from nerif import nerif_match

def func1():
    print("creating new user ...")

def func2():
    print("deleting user ...")

def func3():
    print("making reservation ...")

choice_dict = {
    "func1": "I can create new user in this function",
    "func2": "I can delete user in this function",
    "func3": "I can make reservation for user in this function"
}

match nerif_match.instance(choice_dict, "I wanna use gala server GPU 1-4 tonight"):
    case "func1":
        func1()
    case "func2":
        func2()
    case "func3":
        func3()
