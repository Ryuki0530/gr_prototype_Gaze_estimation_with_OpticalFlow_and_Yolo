import cv2
import tkinter as gui

def connCam(on_device_camera_id = 0):
    result1 = None
    window1 = gui.Tk()
    window1.title("カメラ選択")
    window1.geometry("300x200")

    selectedCamera = gui.StringVar(value="OnDeviceCamera")
    options1 = ["OnDeviceCamera", "IPCamera"]

    for option in options1:
        gui.Radiobutton(window1, text=option, value=option, variable=selectedCamera).pack(anchor=gui.W)

    def onSubmit1():
        nonlocal result1
        result1 = selectedCamera.get()
        window1.destroy()  # GUIを閉じる

    submitButton1 = gui.Button(window1, text="決定", command=onSubmit1)
    submitButton1.pack(pady=10)

    window1.mainloop()

    if result1 == "OnDeviceCamera":
        print("Camera:On This Device Camera")
        return cv2.VideoCapture(on_device_camera_id)

    result2 = None
    window2 = gui.Tk()
    window2.title("IPカメラ選択")
    window2.geometry("300x200")

    label2 = gui.Label(window2, text="IPアドレスを入力してください")
    label2.pack(pady=10)

    ipEntry = gui.Entry(window2, width=30)
    ipEntry.pack(pady=5)