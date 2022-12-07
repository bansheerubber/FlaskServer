import tensorflow as tf
#name = "trained_model_topleft"
#name = "trained_model_bottomleft"
#name = "trained_model_topright"
name = "trained_model_bottomright"
model = tf.keras.models.load_model(name+'.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflmodel = converter.convert()
file = open( name+'.tflite' , 'wb' ) 
file.write( tflmodel )

