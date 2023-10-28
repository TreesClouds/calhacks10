import cvzone, cv2, base64, asyncio
from cvzone.FaceDetectionModule import FaceDetector
from hume import HumeStreamClient, StreamSocket
from hume.models.config import FaceConfig

HUME_KEY = 'A6wdNigFGRQjP7q3616ICffWKxVwpOTTGB60f7IoFnFZAj1R'

def imageCap():
    cap = cv2.VideoCapture(0)
    detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)
    curr_frame_count = 0

    while True:
        _, img = cap.read()
        img, bboxs = detector.findFaces(img, draw=False)

        if bboxs:
            if curr_frame_count == 60:
                _, buffer = cv2.imencode('.jpg', img)
                jpg_as_text = base64.b64encode(buffer)
                curr_frame_count = 0
                break # ! remove this
            curr_frame_count += 1

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return jpg_as_text

async def main():
    client = HumeStreamClient(HUME_KEY)
    config = FaceConfig()
    async with client.connect([config]) as socket:
        # fm = open("example.txt", "r")
        result = await socket.send_bytes(imageCap())    
        emotions = sorted(result['face']['predictions'][0]['emotions'], key=lambda x: x['score'], reverse=True)
        top_emotions = emotions[:5]
        print([i['name'] for i in top_emotions])

asyncio.run(main())