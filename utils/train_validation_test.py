import tensorflow as tf
import tqdm, json

@tf.function
def train(train_ds, model, loss_object, optimizer, metric_objects):
    for images, labels in train_ds:
        # print('images : ', images)
        # print('labels : ', labels)
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object(labels, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        metric_objects['train_loss'](loss)
        metric_objects['train_acc'](labels, predictions)

@tf.function
def validation(validation_ds, model, loss_object, metric_objects):
    for images, labels in validation_ds:
        predictions = model(images)
        loss = loss_object(labels, predictions)

        metric_objects['validation_loss'](loss)
        metric_objects['validation_acc'](labels, predictions)

def test(test_ds, model, loss_object, metric_objects, path_dict):       
    for images, labels in test_ds:
        predictions = model(images)
        loss = loss_object(labels, predictions)
        
        metric_objects['test_loss'](loss)
        metric_objects['test_acc'](labels, predictions)
    
    loss, acc = metric_objects['test_loss'].result().numpy(), metric_objects['test_acc'].result()
    with open(path_dict['cp_path'] + '/test_result.txt', 'w') as wf:
        template = 'test_loss:{}\ntest_acc:{}'
        wf.write(template.format(loss, acc*100))
        print('==== Test Reusult : test_loss:{} / test_acc:{}'.format(loss, acc*100))
