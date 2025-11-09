import cv2
import time
import os
import shutil
import requests
import tkinter as tk
from PIL import Image, ImageTk

# Additional dependencies (install if missing):
# pip install requests pillow
# tkinter is included with most Python installs (on Windows it should be available)
# This script will POST frames to the local FastAPI endpoint at http://127.0.0.1:8000/banana-analysis

def detect_banana(frame):
    """
    Local fallback detection (keeps previous behavior).
    This can be replaced with a local heuristic if desired.
    Return True if a banana is detected in the frame, otherwise False
    """
    return False

API_URL = "http://127.0.0.1:8000/banana-analysis"


def call_api_for_frame(frame, api_url=API_URL, timeout=5):
    """
    Send the provided BGR OpenCV frame to the API as a JPEG multipart upload.
    Returns the parsed JSON response on success, or None on error.
    """
    try:
        ret, buf = cv2.imencode('.jpg', frame)
        if not ret:
            print("Failed to encode frame to JPEG before API call")
            return None

        jpeg_bytes = buf.tobytes()

        files = {
            'file': ('frame.jpg', jpeg_bytes, 'image/jpeg')
        }

        resp = requests.post(api_url, files=files, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"API call failed: {e}")
        return None


def process_frame(frame):
    """
    Called every capture interval. Uses the remote API to decide whether a banana
    is present. Returns a tuple (banana_found: bool, payload: dict|None).
    """
    # Attempt remote API call first
    result = call_api_for_frame(frame)
    banana_found = False
    payload = None

    if result and result.get("message") == "success":
        banana_found = bool(result.get("is_banana", False))
        payload = result
    else:
        # Fallback to local heuristic
        banana_found = detect_banana(frame)
        payload = None

    print("AI processing at", time.strftime("%H:%M:%S"), "| banana_found:", banana_found)
    return banana_found, payload


def show_result_modal(frame, payload=None):
    """
    Show a modal dialog with the captured frame and analysis results.
    Blocks until the user presses the Continue button.
    """
    # Ensure we have a Tk root
    if not hasattr(show_result_modal, "_root") or show_result_modal._root is None:
        root = tk.Tk()
        root.withdraw()
        show_result_modal._root = root
    else:
        root = show_result_modal._root

    # Convert OpenCV BGR frame to PIL Image
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
    except Exception:
        pil_img = None

    top = tk.Toplevel(root)
    top.title("Banana Detected")

    # Image
    if pil_img is not None:
        img_tk = ImageTk.PhotoImage(pil_img.resize((400, 300)))
        img_label = tk.Label(top, image=img_tk)
        img_label.image = img_tk
        img_label.pack()

    # Payload text
    text = tk.Text(top, wrap=tk.WORD, height=10, width=60)
    payload_text = "No details available" if not payload else str(payload)
    text.insert(tk.END, payload_text)
    text.config(state=tk.DISABLED)
    text.pack(padx=8, pady=8)

    done = tk.Event()

    def on_continue():
        top.destroy()

    # Bind keyboard shortcut 'r' or Enter to continue (keyboard fallback)
    def on_key(event):
        # Accept 'r' or Return/Enter
        if event.keysym.lower() == 'r' or event.keysym == 'Return':
            on_continue()

    top.bind('<Key>', on_key)
    top.focus_set()

    btn = tk.Button(top, text="Continue", command=on_continue)
    btn.pack(pady=6)

    # Make the window modal and wait for it to be closed
    top.transient(root)
    top.grab_set()
    root.wait_window(top)


def show_detected_overlay(frame, seconds=2):
    """
    Display the given frame with a prominent "Banana detected!" overlay
    for the specified number of seconds, then return.

    This method freezes the displayed frame (it redraws the same frame repeatedly)
    so the user sees the detection message while the capture loop is paused.
    Pressing 'q' during the overlay will still exit the application.
    """
    overlay = frame.copy()
    h, w = overlay.shape[:2]

    # Draw semi-opaque rectangle background for text
    rect_h = int(h * 0.18)
    cv2.rectangle(overlay, (0, h - rect_h), (w, h), (0, 0, 0), -1)

    # Blend overlay with original for semi-transparent bar
    alpha = 0.6
    blended = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Put text in the bottom center
    text = "BANANA DETECTED!"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(1.0, w / 600)
    thickness = 3
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = int((w - text_size[0]) / 2)
    text_y = h - int(rect_h / 2) + int(text_size[1] / 2)

    cv2.putText(blended, text, (text_x, text_y), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

    # Show the blended frame repeatedly for the given duration
    start = time.time()
    while True:
        cv2.imshow("Live Webcam Feed (press 'q' to quit)", blended)
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            # propagate quit to main loop by raising KeyboardInterrupt
            raise KeyboardInterrupt()
        if time.time() - start >= seconds:
            break

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
    delete_delay = 1.0
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
                banana_found, payload = process_frame(frame)

                #3) If banana detected -> keep and show results, otherwise schedule delete
                if banana_found:
                    print(f"Banana detected; keeping {filename}")
                    # Show a short overlay on the live feed and pause for 2 seconds
                    try:
                        show_detected_overlay(frame, seconds=2)
                    except KeyboardInterrupt:
                        # propagate so outer loop can cleanly exit
                        raise
                    except Exception as e:
                        print(f"Failed to show detection overlay: {e}")
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