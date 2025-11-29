import cv2

VIDEO_PATH = "data/input/video.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

points = []

def callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Clicked: {x}, {y}")

cv2.namedWindow("frame")
cv2.setMouseCallback("frame", callback)

while True:
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cv2.destroyAllWindows()
print("Points:", points)
