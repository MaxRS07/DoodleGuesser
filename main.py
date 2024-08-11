import pygame
import helper
import image
import brain
import numpy as np
WIDTH, HEIGHT = 800, 500

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Doodle Guesser", "DoodleGuesser")
clock = pygame.time.Clock()
open = True
count = 0
drawing = False
pointss = [[]]
font = pygame.font.Font('freesansbold.ttf', 32)
values = [0.0, 0.0, 0.0, 0.0, 0.0]
data = brain.load_train()
model = brain.train(data)
training = False
while open:
    screen.fill("black")
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            open = False
    
    canvas = pygame.draw.rect(screen, "white", (WIDTH / 10, HEIGHT / 8, 3 * HEIGHT / 4, 3 * HEIGHT / 4))
    for i in range(5):
        color = (255,255,255)
        if values[i] == max(values) and values[i] > 0:
            color = (0xfd, 0xdc, 0x5c)
        text = font.render(f"{brain.DoodleClass.all()[i].value}: {values[i] * 100:.2f}%", False, color)
        textRect = text.get_rect()
        textRect.center = (WIDTH // 10 + 3/4 * HEIGHT + textRect.width / 2 + 20, HEIGHT / 2 - 100 + i * 50)
        screen.blit(text, textRect)
        
    mousedown = pygame.mouse.get_pressed()
    mousepos = pygame.mouse.get_pos()
    
    if mousedown[2]:
        pointss.clear()
    if mousedown[0] and helper.contains_point(canvas, mousepos):
        drawing = True
        im = image.create(canvas.width, canvas.height, pointss)
        imdat = helper.to_data(im)
        values = model.predict([imdat])[0]
        if len(pointss[-1]) == 0 or pointss[-1] != mousepos:
            x = mousepos[0] - canvas.left
            y = mousepos[1] - canvas.top
            pointss[-1].append((x,y))
    else:
        drawing = False
        if len(pointss) == 0 or len(pointss[-1]) != 0: 
            pointss.append([])
    
    for points in pointss:
        for i in range(len(points)-1):
            x1 = points[i][0] + canvas.left
            y1 = points[i][1] + canvas.top
            x2 = points[i+1][0] + canvas.left
            y2 = points[i+1][1] + canvas.top
            start = (x1, y1)
            end = (x2, y2)
            pygame.draw.line(screen, "black", start, end, 2)
    
    keys = pygame.key.get_pressed()
    
    if len(pointss) == 0:
            pointss.append([])
            values = [.0,.0,.0,.0,.0]
    
    if keys[pygame.K_ESCAPE]:
        open = False
    if keys[pygame.K_r]:
        data = brain.load_train()
        font.render("Retraining Model...")
        model = brain.train(data)
    import os

    for (i, k) in enumerate([pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5]):
        if keys[k] and len(pointss) > 0 and len(pointss[0]) > 0:
            obj = brain.DoodleClass.all()[i].name
            im = image.create(canvas.width, canvas.height, pointss)
            count = 0
            for (_, _, f) in os.walk(f'Training/{obj}'):
                for i in f:
                    num = int(i[i.index('_')+1:i.index('.')])
                    if num > count:
                        count = num
            im.save(f"Training/{obj}/{obj}_{count + 1}.png")
            pointss.clear()
        
    pygame.display.flip()
    clock.tick(60)
    pygame.display.update()

    