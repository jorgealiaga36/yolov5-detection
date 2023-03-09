import cv2
import torch
from pytube import YouTube


class ObjectDetection:
    """
    Class for implementing Yolov5 model to make inferences on a Youtube video
    """
    def __init__(self,  url, file_name, out_path):
        self.url = url
        self.file_name = file_name
        self.out_path = out_path
        self.model = self.load_model()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self):
        model = torch.hub.load('.\yolov5', 'custom', source='local', path='yolov5s.pt')  # Offline model
        return model

    def get_video(self):
        video = YouTube(self.url).streams.get_highest_resolution()
        video.download(output_path=self.out_path, filename=self.file_name)
        video_root = self.out_path + self.file_name

        return cv2.VideoCapture(video_root)

    def frame_inference(self, frame):
        self.model.to(self.device)
        results = self.model([frame])
        labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

        return labels, cords

    def class_to_label(self, x):
        return self.model.names[int(x)]

    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame





