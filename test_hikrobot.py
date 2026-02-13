import sys
import os
import cv2
import numpy as np

# 1. Point to the DLL location (Change this if your SDK is 32-bit)
mv_dll_path = r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64"
os.add_dll_directory(mv_dll_path)

# 2. Import the Hikrobot wrapper
from MvImport.MvCameraControl_class import *

def main():
    # Initialize SDK and Enumerate Devices
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
    
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0 or deviceList.nDeviceNum == 0:
        print("No devices found!")
        return

    # Create Handle for the first camera
    cam = MvCamera()
    stDeviceImg = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
    ret = cam.MV_CC_CreateHandle(stDeviceImg)
    
    # Open Camera
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    
    # Start Grabbing
    ret = cam.MV_CC_StartGrabbing()

    # Define image buffer size (adjust based on your camera resolution)
    stOutFrame = MV_FRAME_OUT()
    memset(byref(stOutFrame), 0, sizeof(stOutFrame))
    
    # Capture one frame
    ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
    if ret == 0:
        # Convert raw buffer to Numpy array for OpenCV
        # Note: You may need to handle Pixel Format conversion if not Mono8/RGB8
        print(f"Captured frame: {stOutFrame.stFrameInfo.nWidth}x{stOutFrame.stFrameInfo.nHeight}")
        
        # Access pixel data
        pData = (c_ubyte * stOutFrame.stFrameInfo.nFrameLen).from_address(stOutFrame.pBufAddr)
        img = np.frombuffer(pData, dtype=np.uint8).reshape(stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth)
        
        cv2.imshow('Hikrobot Camera', img)
        cv2.waitKey(0)
        
        # Release buffer
        cam.MV_CC_FreeImageBuffer(stOutFrame)

    # Cleanup
    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()

if __name__ == "__main__":
    main()