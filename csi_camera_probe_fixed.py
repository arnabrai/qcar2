'''hardware_test_csi_cameras.py
 
This example demonstrates how to read and display image data
from the 4 csi cameras.
'''
import time
import cv2
from pal.products.qcar import QCarCameras, IS_PHYSICAL_QCAR
import os
from pal.utilities.probe import Probe
 
# Initial Setup
ipHost, ipDriver = '192.168.41.40', 'localhost'
runTime = 120.0 # seconds
counter = 0
 
cameras = QCarCameras(
    enableBack=True,
    enableFront=True,
    enableLeft=True,
    enableRight=True,
)
 
try:
    t0 = time.time()
    probe = Probe(ip = ipHost)
 
    # UPDATED: Wait until probe connects before adding displays
    while not probe.connected:
        probe.check_connection()
        time.sleep(0.1)
 
    # UPDATED: Read one frame first to get correct image size
    flags = cameras.readAll()
    # UPDATED (Solved): Ensure camera read succeeded before accessing imageData
    if all(flags):
        for i, c in enumerate(cameras.csi):
            h, w = c.imageData.shape[:2]
            ch = 3 if len(c.imageData.shape) == 3 else 1
            probe.add_display(
                imageSize=[h, w, ch],  # UPDATED: dynamic size
                scaling=True,
                scalingFactor=2,
                name="CSI"+str(i)
            )
    else:
        print("ERROR: One or more CSI cameras failed to initialize.")
        print("Make sure all QCar2 camera cables are connected properly.")
        raise RuntimeError("CSI camera initialization failed")
 
    # UPDATED (Solved): Fix HTML entity in loop condition
    while time.time() - t0 < runTime:
        if not probe.connected:
            probe.check_connection()
 
        if probe.connected:
            flags = cameras.readAll()
 
            if all(flags):
                counter += 1   # UPDATED: properly inside condition
 
                if counter % 40 == 0:
                    for i, c in enumerate(cameras.csi):
                        probe.send(
                            name="CSI"+str(i),
                            imageData=c.imageData
                        )
            else:
                print("WARNING: Camera frame dropped")
 
except KeyboardInterrupt:
    print('User interrupted.')
 
finally:
    # Termination
    # UPDATED (Solved): Safe termination guards
    try:
        cameras.terminate()
    except:
        pass
    try:
        probe.terminate()
    except:
        pass
