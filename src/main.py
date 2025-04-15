import imageio.v3 as iio
import skimage.transform as ski
import PySimpleGUI as sg
from PIL import Image, ImageTk, ImageSequence

img_files = []
for i in range(1, 8):
    img_files.append("img" + str(i) + ".jpg")

images = [iio.imread(img) for img in img_files]

# Expects the images to be of the same size
iio.imwrite("output.gif", images, duration=0.5, loop=0)

gif_filename = 'output.gif'
layout = [[sg.Image(key='-IMAGE-')]]
window = sg.Window('GIF Output', layout, element_justification='c')

while True:
    for frame in ImageSequence.Iterator(Image.open(gif_filename)):
        event, values = window.read(timeout=100)
        if event == sg.WIN_CLOSED:
            exit(0)
        window['-IMAGE-'].update(data=ImageTk.PhotoImage(frame) )