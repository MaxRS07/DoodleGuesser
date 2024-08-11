import PIL.Image
import PIL.ImageDraw


def create(width: int, height: int, lists: list[list[tuple[int,int]]]):
    image = PIL.Image.new('RGB', (width, height), (255,255,255))
    draw = PIL.ImageDraw.Draw(image)
    
    for list in lists:
        for i in range(len(list)-1):
            draw.line([list[i], list[i + 1]], fill=(0,0,0), width=2)
    
    return image

