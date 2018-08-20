import platform
import sys


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
