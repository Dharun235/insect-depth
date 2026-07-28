import argparse
from pathlib import Path

import cv2


VIDEO_CODEC = "mp4v"


def verify_video(path):
    cap = cv2.VideoCapture(str(path))
    opened = cap.isOpened()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return opened and frame_count > 0


def convert_video(input_path, output_path, overwrite=False):
    if output_path.exists() and not overwrite:
        print(f"Skip existing: {output_path}")
        return True

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Cannot open: {input_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*VIDEO_CODEC),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        print(f"Cannot write: {output_path}")
        return False

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        frame_count += 1

    cap.release()
    writer.release()

    if frame_count == 0 or not verify_video(output_path):
        print(f"Failed verify: {output_path}")
        return False

    print(f"Saved: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert AVI videos to MP4.")
    parser.add_argument("--input-dir", default="data")
    parser.add_argument("--delete-avi", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    avi_files = sorted(input_dir.rglob("*.avi"))

    print(f"Found AVI files: {len(avi_files)}")

    converted = 0
    failed = 0

    for avi_path in avi_files:
        mp4_path = avi_path.with_suffix(".mp4")
        ok = convert_video(avi_path, mp4_path, overwrite=args.overwrite)

        if ok:
            converted += 1
            if args.delete_avi:
                avi_path.unlink()
                print(f"Deleted: {avi_path}")
        else:
            failed += 1

    print(f"Done. Converted/skipped: {converted}. Failed: {failed}.")


if __name__ == "__main__":
    main()
