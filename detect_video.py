from detection_service.servicer import Detect_Service
import argparse
import cv2
import numpy as np
import math

labels_map = {
    0: 'person',
}
colors = np.random.uniform(0, 255, size=(90, 3))

def draw_box_on_frame(frame, det):
    xmin, ymin, xmax, ymax, score, label = int(det[0]), int(det[1]), int(det[2]), int(det[3]), str(det[4]), int(det[5])
    color = colors[label-1]
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.rectangle(frame, (xmin, ymin-30), (xmax, ymin), color, -1)
    cv2.putText(frame, labels_map[label], (xmin+10, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame

def main():
     # ------ Argparse: -------- #
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-model", type=str, required=True)
    parser.add_argument("--inference_url", "-url", type=str, required=True)
    parser.add_argument("--video_path", '-path', type=str, required=True)
    parser.add_argument("--score_threshold", '-score', type=float, default=0.5)
    parser.add_argument("--out_path", '-out', type=str, default='.')
    parser.add_argument("--frame_per_second", '-fps', type=int, default=1)
    args = parser.parse_args()

    detectObj = Detect_Service(args.model_name, args.inference_url, score_threshold=args.score_threshold)

    cap = cv2.VideoCapture(args.video_path)
    in_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = 0
    while True:
        if (idx > 0):
            frames_to_skip = math.ceil(in_fps / args.frame_per_second)
            print(f"Skipping {frames_to_skip} frames")
            idx += frames_to_skip
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if (not ret):
            break
        print(f"Frame: {idx}/{total_frames}")
        detections = detectObj.detect_object(frame)
        for det in detections:
            if int(det[5]) in labels_map:
                frame = draw_box_on_frame(frame, det)

        is_blackout = detectObj.detect_blackout(frame)
        if is_blackout:
            print("Blackout detected!!")

        cv2.imwrite(f"{args.out_path}/frame_{str(idx)}.jpg", frame)
        idx += 1
        if idx >= total_frames:
            break

    cap.release()


if __name__ == '__main__':
    main()