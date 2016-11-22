import math

class Normalizer(object):
    def __init__(self,data):
        self.data = data

    def normalize(self,verbose=False):

        if verbose: print "Normalizing data..."
        normalized = self.__normalize(self.data)

        if verbose: print "=> " + str(normalized)

        if verbose: print "Done!"
        return normalized

    def __normalize(self,data):
        return map(lambda x : (x - self.__mean(data)) / self.__stddev(data),data)

    def __mean(self,array):
        return sum(array) / len(array)

    def __stddev(self,array):
        mean = self.__mean(array)
        variance = reduce(lambda x,y: x + y,map(lambda x : (x - mean) ** 2,array))
        variance /= len(array)
        return math.sqrt(variance)

# ----------------------------  debug  ----------------------------------------
# data = [1,2,7,8,5]
# Normalizer(data).normalize(verbose=True)

#fileWriter: objeto csv.writer para armar el archivo de salida
#imgNum: posicion de la imagen actual en el archivo de test (de 1 a 28000)
#value: valor de la imagen actual que predijo la red neuronal (de 0 a 9)
def writeToSubmissionFile(fileWriter,imgNum, value):
    fileWriter.writerow([imgNum,value])
