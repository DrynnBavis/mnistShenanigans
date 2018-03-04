import network
import mnist_loader

print("Loading MNSIT data...")
#training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
source_data = list(mnist_loader.read("training"))
training_data = source_data[:50000]
validation_data = source_data[50001:60000]
test_data = list(mnist_loader.read("testing"))

print("Creating network...")
net = network.Network([784, 30, 10])
print("Using stochastic grad desc to train...")
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
