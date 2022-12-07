import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# %%
fashion_mnist = tf.keras.datasets.fashion_mnist

'''
    len(train) : 60000
    len(test)  : 10000
    shape : (28, 28)
'''
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# %%
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()



# %%
def test():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    train_x, test_x, train_y, test_y = train_test_split(train_images, train_labels, test_size=0.3)
    model.fit(train_x, train_y, epochs=10, validation_data=(test_x, test_y))

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)


    print("result : ", test_acc)

test()

# %%
"""
    28 X 28
    이미지 4분할
    TL (Top Left) / TR (Top Right) / BL (Bottom Left) / BR (Bottom Right)
    (14, 14), (17, 11), (20, 8), (23, 5)        
"""
def image_split_4():
    upper_images = [i[:23, :] for i in train_images]
    bottom_images = [i[5:, :] for i in train_images]

    TL = [i[:, :23] for i in upper_images]
    TR = [i[:, 5:] for i in upper_images]

    BL = [i[:, :23] for i in bottom_images]
    BR = [i[:, 5:] for i in bottom_images]

    return [np.array(TL), np.array(TR), np.array(BL), np.array(BR)]

split_train_images = image_split_4()

fit = plt.figure()
ax1 = fit.add_subplot(2, 2, 1)
ax1.imshow(split_train_images[0][0])

ax2 = fit.add_subplot(2, 2, 2)
ax2.imshow(split_train_images[1][0])

ax3 = fit.add_subplot(2, 2, 3)
ax3.imshow(split_train_images[2][0])

ax4 = fit.add_subplot(2, 2, 4)
ax4.imshow(split_train_images[3][0])

plt.show()


# %%
def test_image_split_4():
    split_test_images = []
    for i in test_images:
        n = np.random.randint(4)
        if n == 0:
            image = i[:14, :14]
        elif n == 1:
            image = i[:14, 14:]
        elif n == 2:
            image = i[14:, :14]
        else:
            image = i[14:, 14:]

        split_test_images.append(np.array(image))

    return split_test_images

split_test_images = np.array(test_image_split_4())

fit = plt.figure()
image_index = 3
ax1 = fit.add_subplot(2, 2, 1)
ax1.imshow(split_test_images[0])

ax2 = fit.add_subplot(2, 2, 2)
ax2.imshow(split_test_images[1])

ax3 = fit.add_subplot(2, 2, 3)
ax3.imshow(split_test_images[2])

ax4 = fit.add_subplot(2, 2, 4)
ax4.imshow(split_test_images[3])

plt.show()


# %%
"""
    Federated Transfer Learning
"""
# %%
def build_model():
    model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(10)
        ])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    return model

def build_transfer_model(weights=[]):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )


    if len(weights) > 0:
        print("build_transfer_model : set global weight")
        model.set_weights(weights)

    for i, l in enumerate(model.layers):
        if i <= 4:
            l.trainable = False
        else:
            l.trainable = True

    return model

# %%
max_round = 10
local_epochs = 20
fine_epochs = 10
client_num = 4

# %%
'''
    2: 100% transfer, fine tunning 한거 test images로 evaluate
   
    1. global weight -> local training -> fine tuning -> update (1001)
    2. global weight -> fine tuning -> local training -> update (1002)
'''
global_weight = []
round_result = {}

site_1_local = []
site_1_transfer = []
site_1_other = []
site_2_local = []
site_2_transfer = []
site_2_other = []
site_3_local = []
site_3_transfer = []
site_3_other = []
site_4_local = []
site_4_transfer = []
site_4_other = []

for r in range(max_round):
    print("\n========== round : ", r)
    local_model_weight = []

    for i in range(client_num):
        print("\n===== client {} local".format(i))
        train_x, test_x, train_y, test_y = train_test_split(train_images, train_labels, test_size=0.3)

        # fine tuning
        agg_model = build_transfer_model(global_weight)
        # fine tuning
        print("\n===== client {} fine tuning".format(i))
        agg_model.fit(train_x, train_y, epochs=fine_epochs)
        test_loss, test_acc = agg_model.evaluate(test_images, test_labels, verbose=2)

        if i == 0:
            site_1_transfer.append(test_acc)
        elif i == 1:
            site_2_transfer.append(test_acc)
        elif i == 2:
            site_3_transfer.append(test_acc)
        elif i == 3:
            site_4_transfer.append(test_acc)


        # local training
        model = build_model()
        model.set_weights(agg_model.get_weights())

        '''
        if len(global_weight) > 0 :
            print("training set global weights")
            model.set_weights(global_weight)
        '''

        result = model.fit(train_x, train_y, epochs=local_epochs, validation_data=(test_x, test_y))

        test_loss, test_acc = model.evaluate(test_x, test_y, verbose=2)


        if i == 0:
            site_1_local.append(test_acc)
        elif i == 1:
            site_2_local.append(test_acc)
        elif i == 2:
            site_3_local.append(test_acc)
        elif i == 3:
            site_4_local.append(test_acc)

        local_model_weight.append(model.get_weights())


        '''
        # each site
        print("\n===== all test images")
        model = build_model()

        if len(global_weight) > 0:
            model.set_weights(global_weight)
        '''

        #other_loss, other_acc = agg_model.evaluate(split_test_images, test_labels)
        #other_loss, other_acc = agg_model.evaluate(test_images, test_labels)
        other_loss, other_acc = model.evaluate(test_images, test_labels)

        if i == 0:
            site_1_other.append(other_acc)
        elif i == 1:
            site_2_other.append(other_acc)
        elif i == 2:
            site_3_other.append(other_acc)
        elif i == 3:
            site_4_other.append(other_acc)


    print("\n========== weight average")
    # weight avg

    global_weight = []

    for a, b, c, d in zip(local_model_weight[0], local_model_weight[1], local_model_weight[2], local_model_weight[3]):
        weight = (a + b + c + d) / client_num
        global_weight.append(weight)

    save_model = build_model()
    save_model.set_weights(global_weight)
    save_model.save("result/model_{}.h5".format(r))



round_result["site_1_local"] = site_1_local
round_result["site_2_local"] = site_2_local
round_result["site_3_local"] = site_3_local
round_result["site_4_local"] = site_4_local
round_result["site_1_other"] = site_1_other
round_result["site_2_other"] = site_2_other
round_result["site_3_other"] = site_3_other
round_result["site_4_other"] = site_4_other
round_result["site_1_transfer"] = site_1_transfer
round_result["site_2_transfer"] = site_2_transfer
round_result["site_3_transfer"] = site_3_transfer
round_result["site_4_transfer"] = site_4_transfer

df = pd.DataFrame().from_dict(round_result)
df.to_csv("result/result.csv")

print("========== end")


# %%
"""
model = build_model()

# %%
result = model.fit(train_images, 
train_labels, epochs=10)


# %%
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

# %%
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

# %%
predictions_max = np.argmax(predictions, axis=-1)
print(predictions_max)
print(test_labels)

# %%
cm = confusion_matrix(test_labels, predictions_max)
print(cm)

classes = ['T-shirt/Top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
print(classification_report(test_labels, predictions_max, target_names = classes))


fpr = dict()
tpr = dict()
roc_auc = dict()
auprc = dict()
precision = dict()
recall = dict()

#test_labels = np.reshape(test_labels, (-1, 1))
#predictions_max = np.reshape(predictions_max, (-1, 1))

temp = label_binarize(test_labels, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

for i in range(10):
    fpr[i], tpr[i], _ = roc_curve(temp[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    auprc[i] = average_precision_score(temp[:, i], predictions[:, i])
    precision[i], recall[i], _ = precision_recall_curve(temp[:, i], predictions[:, i])

print(roc_auc)

plt.figure()
for i in range(10):
    plt.plot(fpr[i], tpr[i], label='class {0} (area = {1:0.4f})'.format(i, roc_auc[i]))

plt.legend()
plt.show()
"""

# %%
'''
결과 확인
'''
"""
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# %%
# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
"""

# %%

save_model = build_model()
saved_model = tf.keras.models.load_model("result/model_49.h5")
weights = saved_model.get_weights()
weight_list = []
for i in weights:
    weight_list.append(i.tolist())

print(weight_list)