import cv2
import os
import glob


# ==========================
# SETTINGS
# ==========================

INPUT_DIR = "data"
OUTPUT_DIR = os.path.join("output", "video-stable")

os.makedirs(OUTPUT_DIR, exist_ok=True)

VIDEO_EXTENSIONS = ("*.mp4", "*.avi", "*.mov", "*.mkv")


# Feature tracking settings
MAX_CORNERS = 500
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 10

# RANSAC camera estimation
RANSAC_THRESHOLD = 3


# ==========================
# Find videos
# ==========================

videos = []

for ext in VIDEO_EXTENSIONS:
    videos.extend(
        glob.glob(
            os.path.join(INPUT_DIR, "**", ext),
            recursive=True
        )
    )

videos = sorted(videos)

print("Found videos:", len(videos))


# ==========================
# Process each video
# ==========================

# For now, process only the first video.
# Later, change this line to `for video_path in videos:` to process all videos.
for video_path in videos[:1]:

    name = os.path.splitext(
        os.path.basename(video_path)
    )[0]

    output_path = os.path.join(
        OUTPUT_DIR,
        name + "_stable.mp4"
    )

    print("\nProcessing:", video_path)


    cap = cv2.VideoCapture(video_path)

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    ret, prev_frame = cap.read()

    if not ret:
        print("Cannot read video")
        continue


    h, w = prev_frame.shape[:2]


    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w,h)
    )


    # Keep first frame
    writer.write(prev_frame)


    prev_gray = cv2.cvtColor(
        prev_frame,
        cv2.COLOR_BGR2GRAY
    )


    frame_count = 1


    while True:

        ret, curr_frame = cap.read()

        if not ret:
            break


        curr_gray = cv2.cvtColor(
            curr_frame,
            cv2.COLOR_BGR2GRAY
        )


        # ---------------------------------
        # Find feature points in previous frame
        # ---------------------------------

        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=MAX_CORNERS,
            qualityLevel=QUALITY_LEVEL,
            minDistance=MIN_DISTANCE
        )


        if prev_pts is None:

            writer.write(curr_frame)

            prev_gray = curr_gray
            continue



        # ---------------------------------
        # Track points to current frame
        # ---------------------------------

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            prev_pts,
            None
        )


        good_prev = prev_pts[
            status.flatten() == 1
        ]

        good_curr = curr_pts[
            status.flatten() == 1
        ]



        # ---------------------------------
        # Estimate dominant camera motion
        # ---------------------------------

        if len(good_prev) >= 10:

            M, inliers = cv2.estimateAffinePartial2D(
                good_prev,
                good_curr,
                method=cv2.RANSAC,
                ransacReprojThreshold=RANSAC_THRESHOLD
            )

        else:
            M = None



        # ---------------------------------
        # Remove camera motion
        # ---------------------------------

        if M is not None:

            M_inv = cv2.invertAffineTransform(M)

            stabilized = cv2.warpAffine(
                curr_frame,
                M_inv,
                (w,h),
                flags=cv2.INTER_LINEAR
            )

        else:

            stabilized = curr_frame



        writer.write(stabilized)


        prev_gray = curr_gray

        frame_count += 1


        if frame_count % 200 == 0:
            print(
                "Processed frames:",
                frame_count
            )


    cap.release()
    writer.release()


    print("Saved:", output_path)


print("\nDone.")
