# Computer-Vision-Portfolio
A portfolio of 3 chuds in development of computer vision for authentication and security.

---------------Face detection.py---------------

HOW TO USE:

1. Press t to save an instance of yourself, all your face data will be encoded and saved in a file called           reference_encoding.npy.

    Make sure you are within distance of the camera, there will be a indication for this below your face.

    Eye distance will be uses to track how close you are to your face, you must be within this margin because any closer or farther, the code will have trouble distingusihing you from other people. 
    
    Save atleast 5-10 instances, prefferably 10, (very) slight movements and looking at the camera will help alot

2. Then press q, this will save this data and close it

3. Then re-open and now it will test that saved data against your current live-video. 
