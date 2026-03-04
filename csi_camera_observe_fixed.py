from pal.utilities.probe import Observer
# UPDATED (Solved): Added camera import
from pal.products.qcar import QCarCameras     # REQUIRED for CSI images
import time
 
observer = Observer()
 
# UPDATED (Solved): Start observer before adding displays
observer.start()
 
for i in range(4):
    observer.add_display(imageSize = [820,410,3],   # UPDATED (Solved): Corrected image size order (H,W,C)
                        scalingFactor=2,
                        name='CSI'+str(i))
 
observer.launch()
 
# UPDATED (Solved): Added camera initialization
cameras = QCarCameras(
    enableFront=True,
    enableBack=True,
    enableLeft=True,
    enableRight=True
)
 
# UPDATED (Solved): Loop to send camera images to Observer
while True:
    flags = cameras.readAll()
    if all(flags):
        for i, cam in enumerate(cameras.csi):
            observer.send('CSI'+str(i), cam.imageData)
    time.sleep(0.03)
