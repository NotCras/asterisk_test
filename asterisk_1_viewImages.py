import os
import matplotlib.pyplot as pl


if __name__ == "__main__":

    files = [f for f in os.listdir('.') if f[-3:] == 'jpg']

    img = None
    for f in files:
        im = pl.imread(f)
        if img is None:
            img = pl.imshow(im)
        else:
            img.set_data(im)
        pl.pause(.1)
        pl.draw()
