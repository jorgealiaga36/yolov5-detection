import argparse
import cv2
import utls
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', '-u', required=True, type=str, help='video root we want to detect')
    parser.add_argument('--out-source', '-outs', required=True, type=str, help='directory to store output video')
    parser.add_argument('--name', '-n', required=True, type=str, help='output video name')

    return vars(parser.parse_args())


def main():
    """
    url = args['url']
    file_name = args['name']
    out_path = args['out_source']
    """

    url = 'https://www.youtube.com/watch?v=EXUQnLyc3yE&t=14s'
    file_name = 'output_video.mp4'
    out_path = '/data/'

    window_name = "Real Time Object Detection"
    time_init = 0

    detector = utls.ObjectDetection(url, file_name, out_path)
    cap = detector.get_video()

    while True:
        time_new = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        results = detector.frame_inference(frame)
        print(results[0])
        inference = detector.plot_boxes(results, frame)

        fps = 1 / (time_new - time_init)
        time_init = time_new

        cv2.putText(inference, 'fps: ' + str(round(fps)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow(window_name, inference)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # args = parse_args()
    main()
