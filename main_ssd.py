from functions_detection import *
from SSD import process_frame_bgr_with_SSD, get_SSD_model
from vehicle import Vehicle
import os
import os.path as path
import h5py
from moviepy.editor import VideoFileClip

# global deep network model
ssd_model, bbox_helper, color_palette = get_SSD_model()


def process_pipeline(frame, verbose=False):

    detected_vehicles = []

    img_blend_out = frame.copy()

    # return bounding boxes detected by SSD
    ssd_bboxes = process_frame_bgr_with_SSD(frame, ssd_model, bbox_helper, allow_classes=[7], min_confidence=0.3)
    for row in ssd_bboxes:
        label, confidence, x_min, y_min, x_max, y_max = row
        x_min = int(round(x_min * frame.shape[1]))
        y_min = int(round(y_min * frame.shape[0]))
        x_max = int(round(x_max * frame.shape[1]))
        y_max = int(round(y_max * frame.shape[0]))

        proposed_vehicle = Vehicle(x_min, y_min, x_max, y_max)

        if not detected_vehicles:
            detected_vehicles.append(proposed_vehicle)
        else:
            for i, vehicle in enumerate(detected_vehicles):
                if vehicle.contains(*proposed_vehicle.center):
                    pass  # go on, bigger bbox already detected in that position
                elif proposed_vehicle.contains(*vehicle.center):
                    detected_vehicles[i] = proposed_vehicle  # keep the bigger window
                else:
                    detected_vehicles.append(proposed_vehicle)

    # draw bounding boxes of detected vehicles on frame
    for vehicle in detected_vehicles:
        vehicle.draw(img_blend_out, color=(0, 255, 255), thickness=2)

    h, w = frame.shape[:2]
    off_x, off_y = 30, 30
    thumb_h, thumb_w = (96, 128)

    # add a semi-transparent rectangle to highlight thumbnails on the left
    mask = cv2.rectangle(frame.copy(), (0, 0), (w, 2 * off_y + thumb_h), (0, 0, 0), thickness=cv2.FILLED)
    img_blend_out = cv2.addWeighted(src1=mask, alpha=0.3, src2=img_blend_out, beta=0.8, gamma=0)

    # create list of thumbnails s.t. this can be later sorted for drawing
    vehicle_thumbs = []
    for i, vehicle in enumerate(detected_vehicles):
        x_min, y_min, x_max, y_max = vehicle.coords
        vehicle_thumbs.append(frame[y_min:y_max, x_min:x_max, :])

    # draw detected car thumbnails on the top of the frame
    for i, thumbnail in enumerate(sorted(vehicle_thumbs, key=lambda x: np.mean(x), reverse=True)):
        vehicle_thumb = cv2.resize(thumbnail, dsize=(thumb_w, thumb_h))
        start_x = 300 + (i+1) * off_x + i * thumb_w
        img_blend_out[off_y:off_y + thumb_h, start_x:start_x + thumb_w, :] = vehicle_thumb

    # write the counter of cars detected
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_blend_out, 'Vehicles in sight: {:02d}'.format(len(detected_vehicles)),
                (20, off_y + thumb_h // 2), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return img_blend_out

def process_video(input_file, output_file):
    """ Given input_file video, save annotated video to output_file """
    # video = VideoFileClip(input_file).subclip(40,44) # from 38s to 46s
    video = VideoFileClip(input_file)
    annotated_video = video.fl_image(process_pipeline)
    annotated_video.write_videofile(output_file, audio=False)


if __name__ == '__main__':

    mode = 'video'

    if mode == 'video':

        video_file = 'input_video/video1.mov'
        out_path = 'output_video/video1.mp4'

        process_video(video_file, out_path)

        cap_in = cv2.VideoCapture(video_file)

    else:

        test_img_dir = 'test_images'
        for test_img in os.listdir(test_img_dir):

            frame = cv2.imread(os.path.join(test_img_dir, test_img))

            frame_out = process_pipeline(frame, verbose=False)

            cv2.imwrite('output_images/{}'.format(test_img), frame_out)



