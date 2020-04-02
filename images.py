class Image:
    def __init__(self):
        self.__data = {}
        self.__target = -1 #default target

    def GetTarget(self):
        return self.__target

    def SetTarget(self, value):
        self.__target = value

    def SetData(self, value):
        self.__data = value