
import ctypes
import time

SendInput = ctypes.windll.user32.SendInput

W = 0x11
A = 0x1E
S = 0x1F
D = 0x20
L = 0x26
M = 0x32
J = 0x24
P = 0x19  # lock

# C struct redefinitions 
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def attack():
    PressKey(M)
    time.sleep(0.01)
    ReleaseKey(M)

def attack2():
    PressKey(M)
    time.sleep(0.01)
    ReleaseKey(M)

def attack3():
    PressKey(J)
    time.sleep(0.01)
    ReleaseKey(J)

def go_forward():
    PressKey(W)
    time.sleep(0.01)
    ReleaseKey(W)
    
def go_back():
    PressKey(S)
    time.sleep(0.8)
    ReleaseKey(S)
    
def go_left():
    PressKey(A)
    time.sleep(2)
    ReleaseKey(A)
    
def go_right():
    PressKey(D)
    time.sleep(2)
    ReleaseKey(D)
    
def dodge2():  # evade1
    PressKey(L)
    time.sleep(0.01)
    ReleaseKey(L)
    time.sleep(0.01)
    PressKey(L)
    time.sleep(0.01)
    ReleaseKey(L)
'''''
def dodge1():  # evade2
    PressKey(S)
    time.sleep(0.01)
    PressKey(L)
    time.sleep(0.01)
    ReleaseKey(L)
    time.sleep(0.01)
    ReleaseKey(S)
'''

def lock_vision():
    PressKey(P)
    time.sleep(0.01)
    ReleaseKey(P)