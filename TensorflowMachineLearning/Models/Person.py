class Person:
    def __init__(self, nic, name, age):
        self.__nic = str(nic)
        self.__name = str(name)
        self.__age = int(age)

    def get_nic(self):
        return self.__nic

    def get_name(self):
        return self.__name

    def get_age(self):
        return self.__age

