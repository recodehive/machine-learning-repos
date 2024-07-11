import cv2
import detect

import numpy as np
import matplotlib.pyplot as plt


def show_image(img, title='Image'):
    '''
    Displays an image.

    Args:
        img: The image to display.
        title: The title of the window.
    '''

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)

    plt.show()


def mse(img1, img2):
    '''
    Calculates the Mean Squared Error between two images. In short, the lower the value, the more similar the images are.

    Args:
        img1: The first image.
        img2: The second image.
    
    Returns:
        mse: The mean squared error between the two images.
    '''

    h, w = img1.shape
    diff = cv2.subtract(img1, img2)
    err = np.sum(diff ** 2)
    mse = err / (float(h * w))

    return mse


if __name__ == '__main__':
    prev_frame = None
    err = 100
    detected_text = ''

    ERR_DIFF = 20

    vid = cv2.VideoCapture(0)

    try:
        while True:
            _, frame = vid.read()

            if prev_frame is not None:
                err = mse(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

            if err > ERR_DIFF:
                image_data = detect.detect_text(frame)
                detected_text = detect.process_data(frame, image_data)

            cv2.putText(frame, detected_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            prev_frame = frame

            if detected_text != '':
                print(f'Text detected: {detected_text}')
                show_image(frame)

    except KeyboardInterrupt:
        print('Exiting...')
    finally:
        vid.release()