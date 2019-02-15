import cv2
import os
import os.path as path
from SSD import process_frame_bgr_with_SSD, show_SSD_results, get_SSD_model
from moviepy.editor import VideoFileClip

def process_pipeline(frame):
    bboxes = process_frame_bgr_with_SSD(frame, SSD_net, bbox_helper,
                                        min_confidence=0.2,
                                        allow_classes=[2, 7, 14, 15])
    show_SSD_results(bboxes, frame, color_palette=color_palette)
    return frame

def process_video(input_file, output_file):
    """ Given input_file video, save annotated video to output_file """
    # video = VideoFileClip(input_file).subclip(40,44) # from 38s to 46s
    video = VideoFileClip(input_file)
    annotated_video = video.fl_image(process_pipeline)
    annotated_video.write_videofile(output_file, audio=False)

if __name__ == '__main__':
    if not os.path.exists('output_video'):
        os.mkdir('output_video')

    SSD_net, bbox_helper, color_palette = get_SSD_model()

    video_file = 'project_video.mp4'
    out_path = 'project_video_detected.mp4'

    process_video(video_file, out_path)

    # cap = cv2.VideoCapture(video_file)
    #
    # counter = 0
    # while True:
    #
    #     ret, frame = cap.read()
    #
    #     if ret:
    #         bboxes = process_frame_bgr_with_SSD(frame, SSD_net, bbox_helper,
    #                                             min_confidence=0.2,
    #                                             allow_classes=[2, 7, 14, 15])
    #
    #         show_SSD_results(bboxes, frame, color_palette=color_palette)
    #
    #         cv2.imwrite(path.join(out_path, '{:06d}.jpg'.format(counter)), frame)
    #         # cv2.imshow('', frame)
    #         # if cv2.waitKey(1) & 0xFF == ord('q'):
    #         #     break
    #
    #         counter += 1
    #
    #
    # # When everything done, release the capture
    # cap.release()
    # cv2.destroyAllWindows()
    # exit()
