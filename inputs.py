import sched
from time import sleep, time
import win32gui, win32ui, win32con, win32api

# http://www.kbdedit.com/manual/low_level_vk_list.html

LEFT_CTRL = 0xA2
SPACEBAR = 0x20
ENTER = 0x0D


def main():
    # init window hanle
    window_name = "Flappy Bird New"
    hwnd = win32gui.FindWindow(None, window_name)
    hwnds = find_all_windows(window_name)

    # bring each window to the front
    for hwnd in hwnds:
        win32gui.SetForegroundWindow(hwnd)
    sleep(0.01)

    # #start game
    # s = sched.scheduler(time, sleep)
    # for hwnd in hwnds:
    #     # press_key(hwnd, s, LEFT_CTRL, 0.01, 0.1)
    #     # press_key(hwnd, s, ENTER, 0.01, 0.1)
    #     press_key_comb(hwnd, s, LEFT_CTRL, ENTER, 0.01, 0.1)
    #
    #     # offset_secs += 0 # 3.31
    #
    # s.run()

    # play
    for x in range(10):
        s = sched.scheduler(time, sleep)
        offset_secs = 0.35  # 1.0
        for hwnd in hwnds:
            press_key(hwnd, s, SPACEBAR, 0.1 + offset_secs, 0.1)

            # offset_secs += 0 # 3.31

        s.run()


def jump():
    window_name = "Flappy Bird New"
    # hwnds = find_all_windows(window_name)
    # for hwnd in hwnds:
    #     hwnd = win32gui.FindWindow(None, window_name)
    #     win32gui.SetForegroundWindow(hwnd)
    # s = sched.scheduler(time, sleep)
    # for hwnd in hwnds:
    #     press_key(hwnd, s, SPACEBAR, 0.01, 0.03)
    # s.run()
    click(300, 300, "Flappy Bird New")


# send a keyboard input to the given window
def press_key(hwnd, s, key, start_sec, hold_sec):
    priority = 2
    foreground_time = 0.01
    duration = start_sec + hold_sec

    s.enter(start_sec - foreground_time, priority, win32gui.SetForegroundWindow,
            argument=(hwnd,))
    s.enter(start_sec, priority, win32api.SendMessage,
            argument=(hwnd, win32con.WM_KEYDOWN, key, 0))
    s.enter(duration - foreground_time, priority, win32gui.SetForegroundWindow,
            argument=(hwnd,))
    s.enter(duration, priority, win32api.SendMessage,
            argument=(hwnd, win32con.WM_KEYUP, key, 0))


def press_key_comb(hwnd, s, key1, key2, start_sec, hold_sec):
    priority = 2
    foreground_time = 0.15
    duration = start_sec + hold_sec

    s.enter(start_sec - foreground_time, priority, win32gui.SetForegroundWindow,
            argument=(hwnd,))
    s.enter(start_sec, priority, win32api.SendMessage,
            argument=(hwnd, win32con.WM_KEYDOWN, key1, 0))
    s.enter(duration - foreground_time, priority, win32gui.SetForegroundWindow,
            argument=(hwnd,))
    s.enter(start_sec, priority, win32api.SendMessage,
            argument=(hwnd, win32con.WM_KEYDOWN, key2, 0))
    s.enter(duration - foreground_time, priority, win32gui.SetForegroundWindow,
            argument=(hwnd,))
    s.enter(duration, priority, win32api.SendMessage,
            argument=(hwnd, win32con.WM_KEYUP, key1, 0))
    s.enter(duration, priority, win32api.SendMessage,
            argument=(hwnd, win32con.WM_KEYUP, key2, 0))


# win32gui.SetForegroundWindow(hwnd)
# win32api.SendMessage(hwnd, win32con.WM_KEYDOWN, key, 0)
# sleep(sec)
# win32api.SendMessage(hwnd, win32con.WM_KEYUP, key, 0)


def list_window_names():
    def winEnumHandler(hwnd, ctx):
        if win32gui.IsWindowVisible(hwnd):
            print(hex(hwnd), '"' + win32gui.GetWindowText(hwnd) + '"')

    win32gui.EnumWindows(winEnumHandler, None)


def get_inner_windows(whndl):
    def callback(hwnd, hwnds):
        if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
            hwnds[win32gui.GetClassName(hwnd)] = hwnd
        return True

    hwnds = {}
    win32gui.EnumChildWindows(whndl, callback, hwnds)
    return hwnds


def find_all_windows(name):
    result = []

    def winEnumHandler(hwnd, ctx):
        if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd) == name:
            result.append(hwnd)

    win32gui.EnumWindows(winEnumHandler, None)
    return result


def click(x, y, window_name):
    hwnd = win32gui.FindWindow(None, window_name)
    win32gui.SetForegroundWindow(hwnd)

    win32api.SetCursorPos((x, y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)


if __name__ == "__main__":
    while True:
        jump()
        print("jump")
        sleep(0.3)
    # window_name = "Adobe Flash Player 10"
    # hwnd = win32gui.FindWindow(None, window_name)
    # win32gui.SetForegroundWindow(hwnd)
    # click(880, 840)

# list_window_names()
