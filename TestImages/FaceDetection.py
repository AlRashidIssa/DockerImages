import mediapipe as mp
import cv2 as cv
import time

cap = cv.VideoCapture(0)

# Create an instance of the Face Detection module
mpFaceDetection = mp.solutions.face_detection
FaceDetection = mpFaceDetection.FaceDetection()
mpDrawing = mp.solutions.drawing_utils

pTime = 0 
cTime = 0

while True:
    # Read a frame from the video capture object
    ret, img = cap.read()

    # convert image to RGB
    imgRGB = cv.cvtColor(img, code=cv.COLOR_BGR2RGB)
    results = FaceDetection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)

            # Draw bounding box and confidence on the frame
            cv.rectangle(img, bbox, (255, 0, 255), 2)
            cv.putText(img, f'Confidence: {int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # Calculate frames per second (fps)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Display the fps on the frame
    cv.putText(img=img, text=f"FPS: {int(fps)}", org=(10, 50), fontFace=cv.FONT_HERSHEY_SIMPLEX,
               fontScale=1, color=(0, 255, 0), thickness=2)

    # Display the frame in an OpenCV window named "Video"
    cv.imshow("Video", img)

    # Check for the 'q' key press to exit the loop and close the window
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv.destroyAllWindows()