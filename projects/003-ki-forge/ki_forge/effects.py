import cv2
from .particles import Renderer


def hud(frame, energy, fps, demo=False, phase=None):
    h, w = frame.shape[:2]
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (w, 115), (19, 15, 12), -1)
    cv2.rectangle(panel, (0, h-65), (w, h), (19, 15, 12), -1)
    cv2.addWeighted(panel, .78, frame, .22, 0, dst=frame)
    def label(text, pos, size=.5, color=(180, 185, 185)):
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)
    label('KI / FORGE', (28, 42), .95, (240, 245, 250))
    label('HAND-CONTROLLED ENERGY', (30, 66), .4)
    label(f'{"MOUSE LAB" if demo else "LIVE CAMERA"}  /  {fps:.0f} FPS', (w-255, 37), .45)
    label(phase or energy.state, (w-255, 66), .55, (255, 205, 90))
    cv2.rectangle(frame, (30, 85), (w-30, 91), (55, 50, 45), -1)
    cv2.rectangle(frame, (30, 85), (30+int((w-60)*energy.charge), 91), (255, 200, 70), -1)
    instruction = ('MOVE MOUSE: charge   HOLD CLICK: burst' if demo else
                   'OPEN PALM: charge   TWO PALMS: grow   CLOSE FIST: burst')
    label(instruction, (28, h-40), .48, (225, 230, 235))
    label('C  color     M  mute     R  reset     S  screenshot     Q / ESC  quit', (28, h-17), .42)
    return frame
