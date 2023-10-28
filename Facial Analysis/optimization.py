import cv2, base64, asyncio, time, os
from cvzone.FaceDetectionModule import FaceDetector
from hume import HumeStreamClient
from hume.models.config import FaceConfig

HUME_KEY = 'A6wdNigFGRQjP7q3616ICffWKxVwpOTTGB60f7IoFnFZAj1R' # TODO: Gonna have to hide this before we get hacked
# HUME_KEY = os.environ.get('HUME_KEY')

async def main():
    client = HumeStreamClient(HUME_KEY)
    config = FaceConfig()
    async with client.connect([config]) as socket:
        cap = cv2.VideoCapture(0)
        detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)
        curr_frame_count = 0
        curr_second_count = 0
        anger_sum, anxiety_sum, distress_sum = 0, 0, 0
        pain_sum, sadness_sum, tiredness_sum = 0, 0, 0
        
        while True:
            _, img = cap.read()
            img, bboxs = detector.findFaces(img, draw=False)

            if bboxs:
                if curr_frame_count == 60:
                    curr_second_count += 1
                    if curr_second_count > 3: # 3 seconds for the user to get ready
                        _, buffer = cv2.imencode('.jpg', img)
                        jpg_as_text = base64.b64encode(buffer)

                        #TODO: Catch no face error
                        result = await socket.send_bytes(jpg_as_text)   
                        # Catching error for face not detected by Hume
                        if not result['face']['predictions']:
                            continue 
                        scores = result['face']['predictions'][0]['emotions'] 
                        emotions = sorted(result['face']['predictions'][0]['emotions'], key=lambda x: x['score'], reverse=True)
                        top_emotions = emotions[:5]
                        print([i['name'] for i in top_emotions])
                        
                        anger_sum = scores[4]['score']
                        anxiety_sum = scores[5]['score'] 
                        distress_sum = scores[19]['score'] 
                        pain_sum = scores[34]['score'] 
                        sadness_sum = scores[39]['score'] 
                        tiredness_sum = scores[46]['score']
                        
                        if curr_second_count == 8: # 5 seconds for now (assuming 60 FPS) 
                            anger_mean, anxiety_mean, distress_mean = anger_sum / curr_second_count, anxiety_sum / curr_second_count, distress_sum / curr_second_count
                            pain_mean, sadness_mean, tiredness_mean = pain_sum / curr_second_count, sadness_sum / curr_second_count, tiredness_sum / curr_second_count
                            
                            #TODO: Save means in database
                            
                            break
                    curr_frame_count = 0
                curr_frame_count += 1

            cv2.imshow("Image", img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): # TODO: Gonna have to figure out how to handle this later
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
if __name__ == '__main__':
    asyncio.run(main())