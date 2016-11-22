from mlearn import perceptron_network as pn
from mlearn import utils
import csv

dim_input = 784
dim_output = 10
network = pn.Network(dim_input,dim_output,learn_factor=0.4)

#training_stage
train_file = csv.reader(open('train.csv','rb'),delimiter=',')
next(train_file) # skips fields row

count_train = 0
for train in train_file:

    count_train += 1

    train_output = [0,0,0,0,0,0,0,0,0,0]
    train_output[int(train[0])] = 1

    train.pop(0) #removes label
    train_pixels = map(lambda x : float(x),train)
    train_pixels_normalized = utils.Normalizer(train_pixels).normalize()

    network.work(train_pixels_normalized).update(train_output)
    # network.description()

    print "(image_" + str(count_train) + ") Expected:" + str(train_output)
    print "(image_" + str(count_train) + ") Prediction:" + str(network.predict())
    #if count_train == 100: break
del train_file

#testing_stage
test_file = csv.reader(open('test.csv','rb'),delimiter=',')
submission_file = csv.writer(open('submmition.csv','wb'),delimiter=',')
submission_file.writerow(['ImageId','Label'])

next(test_file) # skips fields row
count_test = 0
for test in test_file:

    count_test += 1
    test_pixels = map(lambda x: float(x),test)
    test_pixels_normalized = utils.Normalizer(test_pixels).normalize()

    network.work(test_pixels_normalized)
    prediction = network.predict()

    digit = prediction.index(max(prediction))

    utils.writeToSubmissionFile(submission_file,count_test,digit)

    ##if count_test == 50: break
del test_file
del submission_file
