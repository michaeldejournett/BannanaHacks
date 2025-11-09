import cv2
import time
import os
import shutil

def detect_banana(frame):
    """
    TODO: Replace this with your real AI detection
    Return True is a bana is detected in the frame, otherwise fasle
    """
    return False

def process_frame(frame):
    """
    This function is called every capture interval and runs it through out AI logic.
    """
    banana_found = detect_banana(frame)
    print("AI processing at", time.strftime("%H:%M:%S"), "| banana_found:", banana_found)
    return banana_found

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Fialed to delete {file_path}: {e}")
    else:
        os.makedirs(folder_path)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    last_capture_time = 0 #last time we processed a frame
    capture_interval = 1.0 #seconds between AI scans
    delete_delay = 10.0
    save_dir = "Banana_Pics"

    clear_folder(save_dir)
    print(f"Cleared out existing files in '{save_dir}'")
    
    os.makedirs(save_dir, exist_ok=True)

    pending_delete = {}

    try:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame.")
                break

            now = time.time()

            cv2.imshow("Live Webcam Feed (press 'q' to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User pressed q -- exiting loop.")
                break

            if now - last_capture_time >= capture_interval:
                last_capture_time = now

                #1. saves the current frame
                timestamp = int(now * 1000)
                filename = os.path.join(save_dir, f"frame_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                print(f"Saved frame: {filename}")

                #2. Run AI on the frame
                banana_found = process_frame(frame)

                #3) If no banana-> delete the file we saved
                if banana_found:
                    print(f"Banana detected; keeping {filename}")
                else:
                    pending_delete[filename] = now
                    print(f"No banana; deleting {filename} in {delete_delay} seconds")

            to_delete = [
                f for f, t in pending_delete.items()
                if now - t >= delete_delay
            ]
            for f in to_delete:
                if os.path.exists(f):
                    os.remove(f)
                    print(f"Delete {f} (after {delete_delay}s delay)")
                pending_delete.pop(f, None)
        
    except KeyboardInterrupt:
            print("\nKeyboardInterrupt recieved - exiting cleanly...")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and window closed.")

if __name__ == "__main__":
    main()