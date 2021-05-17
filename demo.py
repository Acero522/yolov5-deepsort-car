from AIDetector_pytorch import Detector
import imutils
import cv2

def main():

    func_status = {}
    func_status['headpose'] = None
    
    name = 'demo'

    det = Detector()
    cap = cv2.VideoCapture('D:\大作业素材\segment-1.avi')
    fps = int(cap.get(5))
    print('fps:', fps)
    t = int(1000/fps) #几毫秒一帧

    size = None
    videoWriter = None

    while True:
        _, im = cap.read()
        if im is None:
            break
        
        result = det.feedCap(im, func_status)
        result = result['frame']
        result = imutils.resize(result, height=500)

        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                'result.mp4', fourcc, fps, (result.shape[1], result.shape[0]))

        videoWriter.write(result)
        cv2.imshow(name, result)
        cv2.waitKey(t)

        if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
            break   # 点x退出

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()