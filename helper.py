import PIL.Image
import pygame
import PIL

def contains_point(rect: pygame.Rect, point: tuple[int, int]) -> bool:
    yrange = point[1] < rect.bottom and point[1] > rect.top
    xrange = point[0] > rect.left and point[0] < rect.right
    if xrange and yrange:
        return True
    return False
def to_data(im: PIL.Image) -> list[float]:
    return [float(abs(int(i[0]) // 255 - 1)) for i in list(im.getdata())]
