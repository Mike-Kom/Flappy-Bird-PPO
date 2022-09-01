import pyautogui
import win32gui
import win32ui
import win32con
import numpy as np
from PIL import Image
import time
import cv2
import pyautogui as pag


def process(screenshot):
    # screenshot = screenshot[30:, 620:1670, :]
    screenshot = cv2.resize(screenshot, (128, 128))
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
    return screenshot / 255.0


def capture_window(windowname="Flappy Bird New", width=640, hight=480, processing=True):
    w = width
    h = hight

    hwnd = win32gui.FindWindow(None, windowname)
    win32gui.SetForegroundWindow(hwnd)
    wDC = win32gui.GetWindowDC(hwnd)
    dcObj = win32ui.CreateDCFromHandle(wDC)
    cDC = dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0, 0), (w, h), dcObj, (0, 0), win32con.SRCCOPY)
    bmpinfo = dataBitMap.GetInfo()
    bmpstr = dataBitMap.GetBitmapBits(True)
    im = Image.frombuffer('RGBA', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), bmpstr, 'raw', 'RGBA', 0, 1)
    screenshot = np.array(im)
    screenshot = screenshot[20:, :, :3]

    if not processing:
        # Free Resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())
        return screenshot
    else:
        screenshot = process(screenshot)
        # Free Resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())
        return screenshot


def game_over(screenshot=None):
    try:
        if not screenshot:
            screenshot = capture_window(processing=False)
    except:
        pass
    if tuple(screenshot[30][320]) == (219, 215, 221):
        return True
    else:
        return False


def main_menu(screenshot=None):
    try:
        if not screenshot:
            screenshot = capture_window(processing=False)
    except:
        pass
    if tuple(screenshot[115][400]) == (74, 56, 82):
        return True
    else:
        return False


if __name__ == "__main__":

    while True:
        time.sleep(0.5)
        if game_over():
            print("Game_over")
        if main_menu():
            print("Main menu")

    # cv2.imshow('screenshot', screenshot)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # screenshot = pyautogui.screenshot("bird.bmp")

    # cv2.imshow('screenshot', screenshot)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # while True:
    #     x, y = pag.position()
    #     r, g, b = pag.pixel(x, y)
    #     print(x, y)
    #     print(r, g, b)
    #     time.sleep(0.5)
