from tqdm import tqdm

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

import os
import requests
import sys
import traceback


def download_model(id, destination):
    '''
    Downloads and saves the model.

    If the model fails to download, it can be downloaded manually:
    https://drive.google.com/file/d/1LAiyCV0p6v-lROdXbtrzKlF4APNmM3Qm/view?usp=sharing
    '''

    URL = "https://docs.google.com/uc?export=download"
    CHUNK_SIZE = 32768

    session = requests.Session()

    response = session.get(URL, params = {'id': id}, stream = True)
    params = {'id': id, 'confirm': 1}
    response = session.get(URL, params = params, stream = True)

    filesize = int(response.headers["Content-Length"])

    try:
        with open(destination, "wb") as f, tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            total=filesize,
            file=sys.stdout,
            desc=destination
        ) as progress:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    datasize = f.write(chunk)
                    progress.update(datasize)
    except BaseException as err:
        traceback.print_exc()
        if os.path.exists(destination): os.remove(destination)
        exit()


if __name__ == '__main__':
    model_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__))) + '\\Model\\Model_Data', 'WordClassifier_Model.h5')

    if not os.path.exists(model_path):
        download_model('1LAiyCV0p6v-lROdXbtrzKlF4APNmM3Qm', model_path)
    
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)
    txt = input("Enter your message: ")

    result_arg = (np.squeeze(model.predict([txt])))
    result = np.argmax(result_arg)
    txt_arg = ["N/A", "Saying Hi", "Borrow Money", "YouTube Interaction"]

    print(f"\nThe prediction is based between - {' | '.join(txt_arg)}.\n")
    print(f"The prediction is {txt_arg[result]}, with the probability of {np.round(result_arg[result]*100, 2)}%")
