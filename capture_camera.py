import cv2
cam_id = 0
FRAME_SIZE = (1920, 1080)
cap = cv2.VideoCapture(cam_id)
ret_width = cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
ret_height = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
assert ret_width and ret_height, 'ret_width:{}, ret_height:{}'.format(ret_width, ret_height)
idx = 0
while True:
    ret, img = cap.read()
    cv2.imshow('preview', img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        output_img = '{}.png'.format(idx)
        print('save file to {}'.format(output_img))
        cv2.imwrite(output_img, img)
        idx += 1
print('done')