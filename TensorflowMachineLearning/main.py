# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from Models.Person import *


def main():
    person = Person('nic001', 'sarala', 23)
    print(person.get_nic())
    print(person.get_name())
    print(person.get_age())


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()




