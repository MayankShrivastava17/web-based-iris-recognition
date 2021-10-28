import gradio as gr
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

title = "IRIS IMAGE RECOGNISATION"

def imageReconization(sepal_length, sepal_width, petal_length, petal_width):
    iris = load_iris()
    knn = KNeighborsClassifier(n_neighbors=1)
    x = iris.data
    y = iris.target
    knn.fit(x, y)
    size = np.array([sepal_length, sepal_width, petal_length, petal_width])
    answer = knn.predict([size])
    if answer == 0:
        return 'setosa.jpg'
    elif answer == 1:
        return'versicolor.jpg'
    elif answer == 2:
        return'viginica.jpg'


iface = gr.Interface(
    fn=imageReconization, 
    inputs=[gr.inputs.Slider(0, 10, 0.1, 0), 
            gr.inputs.Slider(0, 10, 0.1, 0), 
            gr.inputs.Slider(0, 10, 0.1, 0), 
            gr.inputs.Slider(0, 10, 0.1, 0)], 
    outputs=[gr.outputs.Image(label = "IRIS")], 
    title=title
)

iface.launch(share=True)
