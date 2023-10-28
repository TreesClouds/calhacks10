import cv2, base64, asyncio, json, os
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
        
        with open('Facial Analysis\increment_data.json', 'r') as f:
            temp_data = json.load(f)

        while True:
            _, img = cap.read()
            img, bboxs = detector.findFaces(img, draw=False)

            if bboxs:
                if curr_frame_count == 60:
                    # TODO: If we have time, compress image to lower latency
                    _, buffer = cv2.imencode('.jpg', img)
                    jpg_as_text = base64.b64encode(buffer)

                    #TODO: Catch no face error
                    result = await socket.send_bytes(jpg_as_text)    
                    if not result['face']['predictions']:
                        continue
                    scores = result['face']['predictions'][0]['emotions'] 
                    emotions = sorted(result['face']['predictions'][0]['emotions'], key=lambda x: x['score'], reverse=True)
                    top_emotions = emotions[:5]
                    print([i['name'] for i in top_emotions])
                    
                    #TODO: Address resting face of user (user data will be stored in a database)
                    
                    
                    # ? Possible emotions to track: Anxiety, Anger, Distress, Pain, Sadness, Tiredness
                    
                    #TODO: Add emotions that need to be tracked to a 2 minute database (temporary use of json, will shift to Reflex SQL)
                    anger_score = scores[4]['score']
                    anxiety_score = scores[5]['score'] 
                    distress_score = scores[19]['score'] 
                    pain_score = scores[34]['score'] 
                    sadness_score = scores[39]['score'] 
                    tiredness_score = scores[46]['score']
                    
                    #TODO: Address user resting config
                    # 0:Anger, 1:Anxiety, 2:Distress, 3:Pain, 4:Sadness, 5:Tiredness
                    temp_data[curr_second_count] = [anger_score, anxiety_score, distress_score, pain_score, sadness_score, tiredness_score] 
                    with open("Facial Analysis\increment_data.json", "w") as outfile:
                        json.dump(temp_data, outfile)
                    
                    curr_second_count += 1
                    if curr_second_count == 120: # 2 minutes for now (assuming 60 FPS)
                        #TODO: Two minute summary check (take mean of each emotion, go through conditions)
                        anger_sum, anxiety_sum, distress_sum = 0, 0, 0
                        pain_sum, sadness_sum, tiredness_sum = 0, 0, 0
                        
                        for data in temp_data:
                            anger_sum += data[0]
                            anxiety_sum += data[1]
                            distress_sum += data[2]
                            pain_sum += data[3]
                            sadness_sum += data[4]
                            tiredness_sum += data[5]
                            
                        anger_mean, anxiety_mean, distress_mean = anger_sum / curr_second_count, anxiety_sum / curr_second_count, distress_sum / curr_second_count
                        pain_mean, sadness_mean, tiredness_mean = pain_sum / curr_second_count, sadness_sum / curr_second_count, tiredness_sum / curr_second_count
                        
                        # ! Example test case
                        if anger_mean >= 0.8:
                            print("You need to chill out, take a break! *sends browser notification*")
                        
                        # Wipes data from temp database after checking last 2 mins of data
                        temp_data = {}
                        with open("Facial Analysis\increment_data.json", "w") as outfile:
                            json.dump(temp_data, outfile)
                        curr_second_count = 0
                    curr_frame_count = 0
                curr_frame_count += 1

            cv2.imshow("Image", img)

            if cv2.waitKey(1) & 0xFF == ord('q'): # TODO: Gonna have to figure out how to handle this later
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    asyncio.run(main())