import tensorflow as tf
name = "model"
model = tf.keras.models.load_model(name+'.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflmodel = converter.convert()
file = open( name+'.tflite' , 'wb' ) 
file.write( tflmodel )

