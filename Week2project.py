import cv2
import numpy as np

print("🖱️ Click on the ball, then Team 1 jersey, then Team 2 jersey")

hsv_values = []

def click_to_pick(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv = cv2.cvtColor(param, cv2.COLOR_BGR2HSV)
        value = hsv[y, x]
        hsv_values.append(value)
        print("🎯 HSV picked:", value)

cap = cv2.VideoCapture("C:\\Users\\Raghav\\Downloads\\volleyball_match.mp4")
ret, frame = cap.read()
if not ret:
    print("❌ Couldn't read video frame.")
    exit()

cv2.imshow("Click (Ball → Team1 → Team2)", frame)
cv2.setMouseCallback("Click (Ball → Team1 → Team2)", click_to_pick, frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()

if len(hsv_values) < 3:
    print("❌ You must click on: Ball, Team 1 jersey, and Team 2 jersey.")
    exit()

def get_range(hsv, h_margin=10, s_margin=50, v_margin=50):
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
    lower = np.array([max(h - h_margin, 0), max(s - s_margin, 0), max(v - v_margin, 0)], dtype=np.uint8)
    upper = np.array([min(h + h_margin, 179), 255, 255], dtype=np.uint8)
    return lower, upper

ball_lower, ball_upper = get_range(hsv_values[0], 10, 60, 60)
jersey1_lower, jersey1_upper = get_range(hsv_values[1])
jersey2_lower, jersey2_upper = get_range(hsv_values[2])

print("✅ HSV ranges set. Processing video...")

input_path = "C:\\Users\\Raghav\\Downloads\\volleyball_match.mp4"
output_path = "C:\\Users\\Raghav\\Desktop\\volleyball_tracked.avi"

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("❌ Error opening video")
    exit()

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

ball_positions = []
player_counted = False
team1_count, team2_count = 0, 0

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_frame += 1
    if current_frame % 20 == 0:
        print(f"⏳ Processing frame {current_frame}/{frame_count} ({(current_frame/frame_count)*100:.1f}%)")


    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, ball_lower, ball_upper)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 40 < area < 400:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            ball_positions.append(center)
            cv2.circle(frame, center, int(radius), (0, 0, 255), 2)
            break

    for i in range(1, len(ball_positions)):
        cv2.line(frame, ball_positions[i-1], ball_positions[i], (0, 255, 0), 2)

    if not player_counted:
        mask1 = cv2.inRange(hsv, jersey1_lower, jersey1_upper)
        mask2 = cv2.inRange(hsv, jersey2_lower, jersey2_upper)

        contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        team1_count = sum(1 for c in contours1 if cv2.contourArea(c) > 500)
        team2_count = sum(1 for c in contours2 if cv2.contourArea(c) > 500)
        player_counted = True

    cv2.putText(frame, f"Team 1: {team1_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    cv2.putText(frame, f"Team 2: {team2_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print("🎉 DONE! Output saved at:", output_path)

