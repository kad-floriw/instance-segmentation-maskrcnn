import cv2
import json
import logging
import requests
import numpy as np
import matplotlib.pyplot as plt

hostname, port = '127.0.0.1', '5003'
url = 'http://{hostname}:{port}/recognize'.format(hostname=hostname, port=port)


def test_recognize():
    img = cv2.cvtColor(cv2.imread('../train_data/front (1).jpg'), cv2.COLOR_BGR2RGB)
    _, img_encoded = cv2.imencode('.png', img)

    files = {
        'img.png': ('img.png', img_encoded.tostring(), 'image/png')
    }

    response = requests.post(url, files=files)
    result = json.loads(response.text)

    alpha = 0.4
    image_masked = img.copy()
    prediction = result['result'][0]
    for mask, class_id in zip(prediction['masks'], prediction['class_ids']):
        colour = [1., 2., 1.] if class_id == 1 else [2., 1., 1.]
        image_masked[mask[0], mask[1]] = image_masked[mask[0], mask[1]] * colour * alpha + (1 - alpha)

    plt.imshow(np.hstack((img, image_masked)))
    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_recognize()
