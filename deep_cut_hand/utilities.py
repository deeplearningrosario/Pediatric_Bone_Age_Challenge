import platform
import sys
import numpy as np


class Console(object):
    NC = "\033[0m"
    Black = "\033[0;30m"
    DarkGray = "\033[1;30m"
    Red = "\033[0;31m"
    LightRed = "\033[1;31m"
    Green = "\033[0;32m"
    LightGreen = "\033[1;32m"
    BrownOrange = "\033[0;33m"
    Yellow = "\033[1;33m"
    Blue = "\033[0;34m"
    LightBlue = "\033[1;34m"
    Purple = "\033[0;35m"
    LightPurple = "\033[1;35m"
    Cyan = "\033[0;36m"
    LightCyan = "\033[1;36m"
    LightGray = "\033[0;37m"

    def error(*args):
        if platform.system() == "Linux":
            print("[" + Console.Red + "ERROR" + Console.NC + "]", *args)
        else:
            print("[ERROR]", *args)

    def log(*args):
        print(*args)

    def info(*args):
        if platform.system() == "Linux":
            print("[" + Console.Cyan + "INFO" + Console.NC + "]", *args)
        else:
            print("[INFO]", *args)


# Show a progress bar
def updateProgress(progress, tick="", total="", status="Loading..."):
    lineLength = 80
    barLength = 23
    if isinstance(progress, int):
        progress = float(progress)
    if progress < 0:
        progress = 0
        status = "Waiting...\r"
    if progress >= 1:
        progress = 1
        status = ""
    block = int(round(barLength * progress))
    line = str("\rImage: {0}/{1} [{2}] {3}% {4}").format(
        tick,
        total,
        str(("#" * block)) + str("." * (barLength - block)),
        round(progress * 100, 1),
        status,
    )
    emptyBlock = lineLength - len(line)
    emptyBlock = " " * emptyBlock if emptyBlock > 0 else ""
    sys.stdout.write(line + emptyBlock)
    sys.stdout.flush()
    if progress == 1:
        print()


def getHistogram(img):
    hist, _ = np.histogram(img, 256, [0, 256])
    cdf = hist.cumsum()
    return cdf * hist.max() / cdf.max()


# Auto adjust levels colors
# We order the colors of the image with their frequency and
# obtain the accumulated one, then we obtain the colors that
# accumulate 2.5% and 99.4% of the frequency.
def histogramsLevelFix(img, min_color, max_color):
    # This function is only prepared for images in scale of gripes

    # To improve the preform we created a color palette with the new values
    colors_palette = []
    # Auxiliary calculation, avoid doing calculations within the 'for'
    dif_color = 255 / (max_color - min_color)
    for color in range(256):
        if color <= min_color:
            colors_palette.append(0)
        elif color >= max_color:
            colors_palette.append(255)
        else:
            colors_palette.append(int(round((color - min_color) * dif_color)))

    # We paint the image with the new color palette
    height, width = img.shape
    for y in range(0, height):
        for x in range(0, width):
            color = img[y, x]
            img[y, x] = colors_palette[color]

    return img
