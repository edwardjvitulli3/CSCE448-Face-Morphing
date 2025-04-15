import imageio.v3 as iio
import skimage.transform as ski

img_files = []
for i in range(1, 8):
    img_files.append("img" + str(i) + ".jpg")

images = [iio.imread(img) for img in img_files]

# Expects the images to be of the same size
iio.imwrite("output.gif", images, duration=0.5, loop=0)