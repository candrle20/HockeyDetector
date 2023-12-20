import os
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.python.keras.utils.data_utils import get_file


#Set seed for bbox color consistency
np.random.seed(10)

class TFHockeyDetector:
    """Tensor Flow Hockey Detector Class

    Creates a TFHockeyDetector object that detects skaters, refs, and goalies from input video

    Args:
        model_path (str): Path to model file
        output_dir (str): Path to output directory
        class_path (str, optional): Path to file containing class text. Defaults to 'labels.txt'.
    
    Attributes:
        model_name (str): Name of model
        model_dir (str): Path to model directory
        classes (list): List of classes
        classColors (list): List of colors for each class
        model (tf.saved_model): TensorFlow model
    
    Methods:
        obj_classes: Returns list of classes and colors
        load_model: Loads model
        draw_boxes: Draws bounding boxes on image
        image_resize: Resizes image
        detect_video: Detects skaters, refs, and goalies in video
    """
    
    def __init__(self, model_path, output_dir, class_path='labels.txt'):
        
        model_file = os.path.basename(model_path)
        self.model_name =  model_file.split('.')[0]
        
        os.makedirs(self.model_dir, exist_ok=True)
        get_file(model_file, model_path, cache_dir=output_dir, extract=True)
        
        self.classes, self.classColors = self.obj_classes(class_path)
        self.model = self.load_model()


    def obj_classes(self, class_path):
        
        with open(class_path, 'r') as f:
            classes = f.read().splitlines()
        colors = np.random.uniform(0, 255, size=(len(classes), 3))

        return classes, colors


    def load_model(self):
        print('Loading model: ' + self.model_name)
        tf.keras.backend.clear_session()
        model = tf.saved_model.load(os.path.join(self.model_dir, 'datasets', 
                                                      self.model_name, 'saved_model'))
        print('Model loaded successfully!')
        return model


    def draw_boxes(self, image, conf_threshold, thickness=2):
        #Transofrm image to tensor
        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(image)
        inputTensor = inputTensor[tf.newaxis, ...]

        #Predict
        detections = self.model(inputTensor)

        #Get Bounding Box Data
        bboxes = detections['detection_boxes'][0].numpy()
        objClass = detections['detection_classes'][0].numpy().astype(np.int32)
        scores = detections['detection_scores'][0].numpy()

        img_height, img_width, img_channels = image.shape
        frame_boxes = tf.image.non_max_suppression(bboxes, scores, max_output_size=50, 
                                                iou_threshold=0.3, score_threshold=conf_threshold)

        if len(frame_boxes) > 0:
            for i in frame_boxes:
                bbox = tuple(bboxes[i].tolist())
                classConfidence = round(100* scores[i])
                img_class = objClass[i]

                if img_class > 0 and img_class <= len(self.classes):
                    labelText = self.classes[img_class - 1]  # Adjusting index for zero-based list
                    labelColor = self.classColors[img_class - 1]  # Adjusting index for zero-based list

                    #Draw Bounding Box
                    ymin, xmin, ymax, xmax = bbox
                    ymin, xmin, ymax, xmax = int(ymin * img_height), int(xmin * img_width), int(ymax * img_height), int(xmax * img_width)
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), labelColor, thickness=thickness)

                    #Draw Label
                    displayText = f'{labelText}: .{classConfidence}'
                    text_size = cv2.getTextSize(displayText, cv2.FONT_HERSHEY_COMPLEX, 0.4, 1)[0]
                    text_w, text_h = text_size[0], text_size[1]
                    text_x1, text_y1 = xmin, ymin - (text_h + 4)
                    text_x2, text_y2 = xmin + text_w, ymin
                    cv2.rectangle(image, (text_x1, text_y1), (text_x2, text_y2), labelColor, thickness=cv2.FILLED)
                    cv2.putText(image, displayText, (xmin, ymin - 5), cv2.FONT_HERSHEY_COMPLEX, 0.35, (0,0,0), 1)

        return image
    
    
    def detect_video(self, video_path, conf_threshold=0.5, show=True, save=True):
    
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps= cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        current_frame = 0

        base_name = os.path.basename(video_path)
        vid_name, ext = os.path.splitext(base_name)

        if save:
            record = cv2.VideoWriter(f'{vid_name}_bboxed.mp4', cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))

        while True:
            current_frame += 1
            print(f"Processing: {round(current_frame/total_frames * 100, 2)}% complete")
            stream, frame = cap.read()

            if not stream:
                break
            
            box_img = self.draw_boxes(frame, conf_threshold)

            if show:
                cv2.imshow("TensorFlow Resnet50 Hockey Player Detector", box_img)
            if save:
                record.write(box_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        record.release()
        cap.release()
        cv2.destroyAllWindows()


####### ARGS #######
model_dir = '/TF_hockey'
model_path = 'TF_hockeyDetector.h5'
video_path = '/videos/provMich.mp4'


if __name__ == '__main__':
    detector = TFHockeyDetector(model_path, model_dir)
    detector.detect_video(video_path, 0.6)