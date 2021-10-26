import tensorflow as tf
from time import perf_counter
from clearml import Task


task = Task.init(
    project_name='gpu-stress-test',
    task_name='stress test', # task name of at least 3 characters
    task_type=None,
    tags=None,
    reuse_last_task_id=True,
    continue_last_task=False,
    output_uri=None,
    auto_connect_arg_parser=True,
    auto_connect_frameworks=True,
    auto_resource_monitoring=True,
    auto_connect_streams=True,    
)

#download cifar10 dataset
cifar10 = tf.keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

train_set_count = len(train_labels)
test_set_count = len(test_labels)

#setup start time
t1_start = perf_counter()

#normalize images
train_images = train_images / 255.0
test_images = test_images / 255.0

#create ML model using tensorflow provided ResNet50 model, note the [32, 32, 3] shape because that's the shape of cifar
model = tf.keras.applications.ResNet50(
    include_top=True, weights=None, input_tensor=None,
    input_shape=(32, 32, 3), pooling=None, classes=10
)

# CIFAR 10 labels have one integer for each image (between 0 and 10)
# We want to perform a cross entropy which requires a one hot encoded version e.g: [0.0, 0.0, 1.0, 0.0, 0.0...]
train_labels = tf.one_hot(train_labels.reshape(-1), depth=10, axis=-1)

# Do the same thing for the test labels
test_labels = tf.one_hot(test_labels.reshape(-1), depth=10, axis=-1)

#compile ML model, use non sparse version here because there is no sparse data.
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

#train ML model
model.fit(train_images,  train_labels, epochs=10)

#evaluate ML model on test set
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

#setup stop time
t1_stop = perf_counter()
total_time = t1_stop-t1_start

#print results
print('\n')
print(f'Training set contained {train_set_count} images')
print(f'Testing set contained {test_set_count} images')
print(f'Model achieved {test_acc:.2f} testing accuracy')
print(f'Training and testing took {total_time:.2f} seconds')