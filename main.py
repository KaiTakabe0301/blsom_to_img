import numpy as np
import cv2


def convImg2Text(src_path, save_path):
    """
    Convert images to text.
    This function is used to perform BLSOM on an image.
    The converted text is used for principal component analysis in R language and other languages.
    The delimiter is a half-size space.

    Parameters
    ------------
    path : string
        The path of the image to be converted
    """
    img = cv2.imread(src_path)
    height, width, dim = img.shape
    vec = img.reshape((height * width, dim))
    np.savetxt(save_path, vec, delimiter=' ', fmt='%d')


def blsom2Img(path):
    """
    To visualize the map clustering in BLSOM.
    This function can be used when clustering 3D data.
    It is a learning function and is used to check the results of classifying images by color.

    Parameters
    ------------
    path : string
        Maps learned in BLSOM
    """
    with open(path) as f:
        dim = int(f.readline().strip())
        width = int(f.readline().strip())
        height = int(f.readline().strip())

        img = np.empty(shape=(height, width, dim), dtype=np.uint8)
        for h in range(0, height):
            row = np.empty(shape=(width, dim), dtype=np.uint8)
            for w in range(0, width):
                line = f.readline().strip()
                if (len(line) > 0):
                    pixel = line.replace('\n', '').split(' ')[:3]
                    pixel = list(map(lambda x: round(float(x)), pixel))
                    pixel = np.array(pixel, dtype=np.uint8)
                    row[w] = pixel
            img[h] = row

        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


convImg2Text('sample/convImg2Text/sample001.png',
             'sample/convImg2Text/convImg2Txt.txt')
blsom2Img('sample/blsom2Img/result_batch.txt')
