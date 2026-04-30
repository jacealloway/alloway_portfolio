import numpy as np
import pygame
import sys 
from time import time


## ALGORITHM FUNCTIONS

def cellBlanks(grid: list) -> list:
    """
    Find all empty spaces in a grid symbolized by a ".".

    **Returns:** *list*. Cell indices of all empty entries of the input grid.
    """
    cell_history = []
    grid = np.array(grid)
    n, m = np.shape(grid)
    for i in range(n):
        for j in range(m):
            entry = grid[i][j]
            if entry == ".":
                cell_history.append([i, j])
    return cell_history 


def boxes(grid: list) -> list: 
    """
    Transforms the shape of the input grid (3x3)x(3x3) into their respectibe boxes (1x9)x(3x3).

    **Returns:** *list*. Transformed shape of input grid.
    """
    grid = np.array(grid)
    empty = np.zeros(grid.shape, dtype=str)
    empty = empty.reshape(9, 3, 3)
        #fill the empty cells with grid nums
    box_num, box_row = empty.shape[0:2]
    for l in range(box_num):
        for m in range(box_row):
            empty[l][m] = grid[m + 3*int(np.floor(l/3))][(l%3)*3:(l%3)*3+3]
    return empty


def columns(grid: list) -> list:
    """
    Transforms the shape of the input grid (3x3)x(3x3) into their respectibe columns (3x3)x(3x3).

    **Returns:** *list*. Transformed shape of input grid. Obtains by taking the transpose of the row-format input grid.
    """
    grid = np.array(grid)
    return np.transpose(grid)


def validator(grid: list, index: list) -> bool:
    """
    Determine the validity of the cell entry.

    **Returns:** *bool*. If entry is valid compared to surrounding boxes/rows/columns, return True. Else return False.
    """
    grid = np.array(grid)
    i = index[0]                #the row value
    j = index[1]                #the column value
    num_check = grid[i][j]      #extract the value of the cell entry
    check_rows = False
    check_cols = False
    check_box = False
    
        #1. check rows
    if num_check in np.delete(grid[i], j):
        return False 
    else:
        check_rows = True
        
        #2. check columns
    if num_check in np.delete(columns(grid)[j], i):
        return False
    else:
        check_cols = True
        
        #3. check boxes
    ijpbox = int(np.floor((j+9*np.floor(i/3))/3))  #the box number 
    iprime = i%3
    jprime = j%3
    checker = []
    
    for boxrow in range(3):
        checker = np.append(checker, (boxes(grid)[ijpbox][boxrow]))
        idxmap = int(3*iprime + jprime)

    if num_check in np.delete(checker, idxmap):
        return False
            
    else:
        check_box = True
            
    if check_rows==True and check_cols==True and check_box==True:
        return True


def gridMemory(grid: list) -> list:
    """
    Produce a replication of the input grid empty cells so that cell attempts may be stored in a memory.

    **Returns:** *list*. For each "." entry in input grid, copy position and append to array to be filled with potential values in a new list whose shape is identical to input grid. If input grid entry is valued already, memory entry returns 0 for that cell.
    """
    memory_grid = np.zeros((np.shape(grid)), dtype = object)    #set dtype = object to store history arrays (we don't need to access these arrays, we just need to append them so that we can do a numpy.delete() on the temp nums-fill-list)
    n, m = np.shape(grid)
    for i in np.arange(n):
        for j in np.arange(m):
            if grid[i][j] == ".":
                memory_grid[i][j] = np.array(["."])

    return memory_grid
    



def_grid = [["5","3",".",".","7",".",".",".","."],
            ["6",".",".","1","9","5",".",".","."],
            [".","9","8",".",".",".",".","6","."],
            ["8",".",".",".","6",".",".",".","3"],
            ["4",".",".","8",".","3",".",".","1"],
            ["7",".",".",".","2",".",".",".","6"],
            [".","6",".",".",".",".","2","8","."],
            [".",".",".","4","1","9",".",".","5"],
            [".",".",".",".","8",".",".","7","9"]]

empty_grid = [[".",".",".",".",".",".",".",".","."],
            [".",".",".",".",".",".",".",".","."],
            [".",".",".",".",".",".",".",".","."],
            [".",".",".",".",".",".",".",".","."],
            [".",".",".",".",".",".",".",".","."],
            [".",".",".",".",".",".",".",".","."],
            [".",".",".",".",".",".",".",".","."],
            [".",".",".",".",".",".",".",".","."],
            [".",".",".",".",".",".",".",".","."]] 





#WRITE INTERFACE
pygame.init()    
    #starting variables
width = 600         #width of window
height = 730        #height of window  
screen = pygame.display.set_mode((width, height))   #make screen
screen.fill('White')                                #set background color
pygame.display.set_caption('Sudoku Solver')         #set window name
clock = pygame.time.Clock()         #set framerate

font1 = pygame.font.SysFont('timesnewroman', 40)      #set font
font2 = pygame.font.SysFont('timesnewroman', 25) 
font3 = pygame.font.SysFont('timesnewroman', 45)
font4 = pygame.font.SysFont('timesnewroman', 50)
title = font1.render("Sudoku Solver", True, 'Black')
runlabel = font2.render("Run (R)", True, 'Black')
clearlabel = font2.render("Clear Grid (C)", True, 'Black')
resetlabel = font2.render("Load Last Input (L)", True, 'Black')
deflabel = font2.render("Load Example Grid (E)", True, 'Black')
timelabel = font2.render("Time: ", True, 'Black')
one = font3.render("1", True, 'Black')
two = font3.render("2", True, 'Black')
three = font3.render("3", True, 'Black')
four = font3.render("4", True, 'Black')
five = font3.render("5", True, 'Black')
six = font3.render("6", True, 'Black')
seven = font3.render("7", True, 'Black')
eight = font3.render("8", True, 'Black')
nine = font3.render("9", True, 'Black') 
clearwhite= font4.render('■', True, 'White')   



start = 50
end = 550
linespace = np.linspace(start+3, end-3, 10)         #width spacing of input grid
widths = (8, 2, 2, 5, 2, 2, 5, 2, 2, 8)             #set the line spacings
gridspace = np.linspace(start+3, end-3, 10)         #grid cells
griddifference = np.round(gridspace[1] - gridspace[0], 2)
gridin = np.empty((9,9), dtype = object)           #determine the locations of the upper-right grid edges; index with i,j
gridin.fill(".")                                   #create empty initialized input grid


def get_cell(pos: list[float]) -> list[float]:
    """
    Get current cell index of input pixel position.
    """
    xind = pos[0]
    yind = pos[1]
    
    cell_i = np.floor(xind/griddifference)
    cell_j = np.floor(yind/griddifference)
    return (int(cell_i) - 1, int(cell_j) - 1)




    #set default selector position


i, j = (0, 0)
Run = False
complete = False
clear_grid = False
reset_grid = False
defaultgrid = False


while True: 
    pygame.draw.polygon(screen, 'Black', ((70, 570), (100, 585), (70, 600)))
    pygame.draw.polygon(screen, '#a7ff36', ((72, 573), (96, 585), (72, 597)))   
    pygame.draw.polygon(screen, '#a7ff36', ((gridspace[i], gridspace[j]), (gridspace[i], gridspace[j] + griddifference), (gridspace[i] + griddifference, gridspace[j] + griddifference), (gridspace[i] + griddifference, gridspace[j])))


    if complete == True:
        Run = False


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            if 53 < pos[0] < 547 and 53 < pos[1] < 547:
                i, j = get_cell(pos)
                pygame.draw.polygon(screen, 'White', ((gridspace[0], gridspace[0]), (gridspace[0], gridspace[9]), (gridspace[9], gridspace[9]), (gridspace[9], gridspace[0])))
                pygame.draw.polygon(screen, '#a7ff36', ((gridspace[i], gridspace[j]), (gridspace[i], gridspace[j] + griddifference), (gridspace[i] + griddifference, gridspace[j] + griddifference), (gridspace[i] + griddifference, gridspace[j])))
    
            
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                i = i
                j = j-1
                if j == -1:
                    j = 8
                pygame.draw.polygon(screen, 'White', ((gridspace[0], gridspace[0]), (gridspace[0], gridspace[9]), (gridspace[9], gridspace[9]), (gridspace[9], gridspace[0])))
                pygame.draw.polygon(screen, '#a7ff36', ((gridspace[i], gridspace[j]), (gridspace[i], gridspace[j] + griddifference), (gridspace[i] + griddifference, gridspace[j] + griddifference), (gridspace[i] + griddifference, gridspace[j])))
                
            if event.key == pygame.K_DOWN:
                i = i
                j = j+1
                if j == 9:
                    j = 0
                pygame.draw.polygon(screen, 'White', ((gridspace[0], gridspace[0]), (gridspace[0], gridspace[9]), (gridspace[9], gridspace[9]), (gridspace[9], gridspace[0])))
                pygame.draw.polygon(screen, '#a7ff36', ((gridspace[i], gridspace[j]), (gridspace[i], gridspace[j] + griddifference), (gridspace[i] + griddifference, gridspace[j] + griddifference), (gridspace[i] + griddifference, gridspace[j])))
                
            if event.key == pygame.K_LEFT:
                i = i-1
                j = j
                if i == -1:
                    i = 8
                pygame.draw.polygon(screen, 'White', ((gridspace[0], gridspace[0]), (gridspace[0], gridspace[9]), (gridspace[9], gridspace[9]), (gridspace[9], gridspace[0])))
                pygame.draw.polygon(screen, '#a7ff36', ((gridspace[i], gridspace[j]), (gridspace[i], gridspace[j] + griddifference), (gridspace[i] + griddifference, gridspace[j] + griddifference), (gridspace[i] + griddifference, gridspace[j])))
                
            if event.key == pygame.K_RIGHT:
                i = i+1
                j = j
                if i == 9:
                    i = 0
                pygame.draw.polygon(screen, 'White', ((gridspace[0], gridspace[0]), (gridspace[0], gridspace[9]), (gridspace[9], gridspace[9]), (gridspace[9], gridspace[0])))
                pygame.draw.polygon(screen, '#a7ff36', ((gridspace[i], gridspace[j]), (gridspace[i], gridspace[j] + griddifference), (gridspace[i] + griddifference, gridspace[j] + griddifference), (gridspace[i] + griddifference, gridspace[j])))

            if event.key == pygame.K_SPACE:
                gridin[i][j] = "0"
            if event.key == pygame.K_1:
                gridin[i][j] = "1"
            if event.key == pygame.K_2:
                gridin[i][j] = "2"
            if event.key == pygame.K_3:
                gridin[i][j] = "3"
            if event.key == pygame.K_4:
                gridin[i][j] = "4"
            if event.key == pygame.K_5:
                gridin[i][j] = "5"
            if event.key == pygame.K_6:
                gridin[i][j] = "6"
            if event.key == pygame.K_7:
                gridin[i][j] = "7"
            if event.key == pygame.K_8:
                gridin[i][j] = "8"
            if event.key == pygame.K_9:
                gridin[i][j] = "9"

            if event.key == pygame.K_c:
                clear_grid = True
            if event.key == pygame.K_l:
                reset_grid = True
            if event.key == pygame.K_e:
                defaultgrid = True

                
                



    if clear_grid == True:
        gridin = np.copy(empty_grid)
        clear_grid = False
        complete = False
        pygame.draw.polygon(screen, 'White', ((gridspace[0], gridspace[0]), (gridspace[0], gridspace[9]), (gridspace[9], gridspace[9]), (gridspace[9], gridspace[0])))

    if reset_grid == True:
        gridin = np.copy(GRID_INIT)
        reset_grid = False
        complete = False
        pygame.draw.polygon(screen, 'White', ((gridspace[0], gridspace[0]), (gridspace[0], gridspace[9]), (gridspace[9], gridspace[9]), (gridspace[9], gridspace[0])))

    if defaultgrid == True:
        gridin = np.copy(def_grid)
        defaultgrid = False
        complete = False
        pygame.draw.polygon(screen, 'White', ((gridspace[0], gridspace[0]), (gridspace[0], gridspace[9]), (gridspace[9], gridspace[9]), (gridspace[9], gridspace[0])))



    for n in range(9):
        for m in range(9):
            if gridin[n][m] == "1":
                pygame.draw.polygon(screen, '#a7ff36', ((gridspace[n], gridspace[m]), (gridspace[n], gridspace[m] + griddifference), (gridspace[n] + griddifference, gridspace[m] + griddifference), (gridspace[n] + griddifference, gridspace[m])))
                screen.blit(one, (gridspace[n]+18, gridspace[m]+3))
            if gridin[n][m] == "2":
                pygame.draw.polygon(screen, '#a7ff36', ((gridspace[n], gridspace[m]), (gridspace[n], gridspace[m] + griddifference), (gridspace[n] + griddifference, gridspace[m] + griddifference), (gridspace[n] + griddifference, gridspace[m])))
                screen.blit(two, (gridspace[n]+18, gridspace[m]+3))
            if gridin[n][m] == "3":
                pygame.draw.polygon(screen, '#a7ff36', ((gridspace[n], gridspace[m]), (gridspace[n], gridspace[m] + griddifference), (gridspace[n] + griddifference, gridspace[m] + griddifference), (gridspace[n] + griddifference, gridspace[m])))
                screen.blit(three, (gridspace[n]+18, gridspace[m]+3))
            if gridin[n][m] == "4":
                pygame.draw.polygon(screen, '#a7ff36', ((gridspace[n], gridspace[m]), (gridspace[n], gridspace[m] + griddifference), (gridspace[n] + griddifference, gridspace[m] + griddifference), (gridspace[n] + griddifference, gridspace[m])))
                screen.blit(four, (gridspace[n]+18, gridspace[m]+3))
            if gridin[n][m] == "5":
                pygame.draw.polygon(screen, '#a7ff36', ((gridspace[n], gridspace[m]), (gridspace[n], gridspace[m] + griddifference), (gridspace[n] + griddifference, gridspace[m] + griddifference), (gridspace[n] + griddifference, gridspace[m])))
                screen.blit(five, (gridspace[n]+18, gridspace[m]+3))
            if gridin[n][m] == "6":
                pygame.draw.polygon(screen, '#a7ff36', ((gridspace[n], gridspace[m]), (gridspace[n], gridspace[m] + griddifference), (gridspace[n] + griddifference, gridspace[m] + griddifference), (gridspace[n] + griddifference, gridspace[m])))
                screen.blit(six, (gridspace[n]+18, gridspace[m]+3))
            if gridin[n][m] == "7":
                pygame.draw.polygon(screen, '#a7ff36', ((gridspace[n], gridspace[m]), (gridspace[n], gridspace[m] + griddifference), (gridspace[n] + griddifference, gridspace[m] + griddifference), (gridspace[n] + griddifference, gridspace[m])))
                screen.blit(seven, (gridspace[n]+18, gridspace[m]+3))
            if gridin[n][m] == "8":
                pygame.draw.polygon(screen, '#a7ff36', ((gridspace[n], gridspace[m]), (gridspace[n], gridspace[m] + griddifference), (gridspace[n] + griddifference, gridspace[m] + griddifference), (gridspace[n] + griddifference, gridspace[m])))
                screen.blit(eight, (gridspace[n]+18, gridspace[m]+3))
            if gridin[n][m] == "9":
                pygame.draw.polygon(screen, '#a7ff36', ((gridspace[n], gridspace[m]), (gridspace[n], gridspace[m] + griddifference), (gridspace[n] + griddifference, gridspace[m] + griddifference), (gridspace[n] + griddifference, gridspace[m])))
                screen.blit(nine, (gridspace[n]+18, gridspace[m]+3))
            if gridin[n][m] == "0":
                pygame.draw.polygon(screen, '#a7ff36', ((gridspace[n], gridspace[m]), (gridspace[n], gridspace[m] + griddifference), (gridspace[n] + griddifference, gridspace[m] + griddifference), (gridspace[n] + griddifference, gridspace[m])))
                gridin[n][m] = "."


    for k in range(len(linespace)):                 #draw the gridlines - last....
        pygame.draw.line(screen, 'Black', (linespace[k], start), (linespace[k], end), width = widths[k])
        pygame.draw.line(screen, 'Black', (start, linespace[k]), (end, linespace[k]), width = widths[k])
    
    

    click = pygame.mouse.get_pressed()[0]
    pos = pygame.mouse.get_pos()
    if 70 < pos[0] < 100 and 570 < pos[1] < 600:
        if click:
            pygame.draw.polygon(screen, '#80cc1f', ((72, 573), (96, 585), (72, 597)))
            Run = True
    keys = pygame.key.get_pressed()
    if keys[pygame.K_r]:
        pygame.draw.polygon(screen, '#80cc1f', ((72, 573), (96, 585), (72, 597)))
        Run = True

    #DRAW LABELS
    screen.blit(title, (185, 7))
    screen.blit(runlabel, (110, 570))
    screen.blit(clearlabel, (110, 620))
    screen.blit(resetlabel, (310, 570))
    screen.blit(deflabel, (310, 620))
    screen.blit(timelabel, (72, 670))


    #re-draw the highlighting grid border
    pygame.draw.polygon(screen, 'Black', ((gridspace[i]+2, gridspace[j]+2), (gridspace[i]+2, gridspace[j] + griddifference - 1), (gridspace[i] + griddifference - 1, gridspace[j] + griddifference - 1), (gridspace[i] + griddifference - 1, gridspace[j] + 2)), width = 3)


    pygame.display.update()

    if Run:
            #initial vars / conditions 
        nums_fill = np.array([".", "1", "2", "3", "4", "5", "6", "7", "8", "9"])    #possible entries in each cell    
        gridEdit = gridin                                                           #copy the input fullgrid array to make a new editing grid
        GRID_INIT = np.copy(gridin)
        memory = gridMemory(gridEdit)                                               #generate an empty memory grid for previous entries
        cell_history = cellBlanks(gridEdit)                                         #generate all the entries of the empty spaces 
        counter = 0                                                                 #initialize counter variable for cell history indexing
        t1 = time()                                                                 #time the solver start as t1           
        countHist = []                                                              #initialize counter history

            #make black border go away
        if gridEdit[i][j] == ".":
            pygame.draw.polygon(screen, 'White', ((gridspace[i], gridspace[j]), (gridspace[i], gridspace[j] + griddifference), (gridspace[i] + griddifference, gridspace[j] + griddifference), (gridspace[i] + griddifference, gridspace[j])))

        if gridEdit[i][j] != ".":
            pygame.draw.polygon(screen, 'White', ((gridspace[i], gridspace[j]), (gridspace[i], gridspace[j] + griddifference), (gridspace[i] + griddifference, gridspace[j] + griddifference), (gridspace[i] + griddifference, gridspace[j])))
            pygame.draw.polygon(screen, '#a7ff36', ((gridspace[i], gridspace[j]), (gridspace[i], gridspace[j] + griddifference), (gridspace[i] + griddifference, gridspace[j] + griddifference), (gridspace[i] + griddifference, gridspace[j])))


        while counter < len(cell_history):                                  #while the counter is less than the possible empty positions, continue the loop. Else break it.    
            er, ec = cell_history[counter]                                  #find the index of the blank space as function of index counter
            grid_nums = np.array(list(set(nums_fill) - set(memory[er][ec])))#create the grid attempt list to only iterate through possible numbers provided by validator 
            grid_nums= np.sort(grid_nums)                                   #sort the list so you have the same entry order each time   

            countHist.append(counter)

            pygame.draw.polygon(screen, '#75eaff', ((gridspace[er], gridspace[ec]), (gridspace[er], gridspace[ec] + griddifference), (gridspace[er] + griddifference, gridspace[ec] + griddifference), (gridspace[er] + griddifference, gridspace[ec])))
            for k in range(len(linespace)):                 #draw the gridlines again
                pygame.draw.line(screen, 'Black', (linespace[k], start), (linespace[k], end), width = widths[k])
                pygame.draw.line(screen, 'Black', (start, linespace[k]), (end, linespace[k]), width = widths[k])

            for n in range(9):
                for m in range(9):
                    if gridEdit[n][m] == "1":
                        screen.blit(one, (gridspace[n]+18, gridspace[m]+3))
                    if gridEdit[n][m] == "2":
                        screen.blit(two, (gridspace[n]+18, gridspace[m]+3))
                    if gridEdit[n][m] == "3":
                        screen.blit(three, (gridspace[n]+18, gridspace[m]+3))
                    if gridEdit[n][m] == "4":
                        screen.blit(four, (gridspace[n]+18, gridspace[m]+3))
                    if gridEdit[n][m] == "5":
                        screen.blit(five, (gridspace[n]+18, gridspace[m]+3))
                    if gridEdit[n][m] == "6":
                        screen.blit(six, (gridspace[n]+18, gridspace[m]+3))
                    if gridEdit[n][m] == "7":
                        screen.blit(seven, (gridspace[n]+18, gridspace[m]+3))
                    if gridEdit[n][m] == "8":
                        screen.blit(eight, (gridspace[n]+18, gridspace[m]+3))
                    if gridEdit[n][m] == "9":
                        screen.blit(nine, (gridspace[n]+18, gridspace[m]+3))
                    if gridEdit[n][m] == "." and er != n and ec != m:
                        screen.blit(clearwhite, (gridspace[n]+18, gridspace[m]+3))

            pygame.display.update()         #update everything to screen


            if len(memory[er][ec]) == 10:                       #if you've attempted all the numbers in a cell already, step back, erase history
                memory[er][ec] = ["."]                          #overwrite entries as blank       
                gridEdit[er][ec] = "."

                pygame.draw.polygon(screen, 'White', ((gridspace[er], gridspace[ec]), (gridspace[er], gridspace[ec] + griddifference), (gridspace[er] + griddifference, gridspace[ec] + griddifference), (gridspace[er] + griddifference, gridspace[ec])))

                counter = counter - 1                           #step back
                
                if counter == -1:                               #ensure non-negative counter
                    counter = 0

                er_prev, ec_prev = cell_history[counter]        #determine the index of the last entry
                gridEdit[er_prev][ec_prev] = "."                #overwrite previous entry as blank as well; do not record history  
                
                pygame.draw.polygon(screen, 'White', ((gridspace[er], gridspace[ec]), (gridspace[er], gridspace[ec] + griddifference), (gridspace[er] + griddifference, gridspace[ec] + griddifference), (gridspace[er] + griddifference, gridspace[ec])))


            for num in grid_nums:                                   #loop over the grid nums list                     
                gridEdit[er][ec] = num                              #fill the grid cell with the number 
                valid = validator(gridEdit, [er, ec])               #determine the validity of the entry 
                memory[er][ec] = np.append(memory[er][ec], num)     #record entry history

                if valid == True:                                   #IF TRUE: step forward

                    pygame.draw.polygon(screen, 'White', ((gridspace[er], gridspace[ec]), (gridspace[er], gridspace[ec] + griddifference), (gridspace[er] + griddifference, gridspace[ec] + griddifference), (gridspace[er] + griddifference, gridspace[ec])))

                    counter = counter + 1                           
                    break
                
                if valid == False and len(memory[er][ec]) == 10:    #IF TRUE and all nums have been attempted for that cell, step back 
                    gridEdit[er][ec] = "."                          #overwrite grid value with empty
                    memory[er][ec] = np.array(["."])                #empty the memory array for grid cell 
                    
                    pygame.draw.polygon(screen, 'White', ((gridspace[er], gridspace[ec]), (gridspace[er], gridspace[ec] + griddifference), (gridspace[er] + griddifference, gridspace[ec] + griddifference), (gridspace[er] + griddifference, gridspace[ec])))
                    
                    counter = counter - 1                           #step back 

                    if counter == -1:                               #ensure non-negative counter
                        counter = 0
                    break                                           #break for-loop if statement satisfied

                if valid == False:                                  #IF TRUE and you haven't gone thru all nums in list yet: stay
                    pass   

                    pygame.draw.polygon(screen, 'White', ((gridspace[er], gridspace[ec]), (gridspace[er], gridspace[ec] + griddifference), (gridspace[er] + griddifference, gridspace[ec] + griddifference), (gridspace[er] + griddifference, gridspace[ec])))
            
                t2 = time()

                time_counter = font2.render('{}'.format(np.round((t2 - t1), 3)), True, 'Black')
                pygame.draw.polygon(screen, 'White', ((130, 650), (400, 650), (400, 710), (130, 710)))
                screen.blit(time_counter, (140, 670))

        returnGrid = np.copy(gridEdit)
        complete = True  






 
    pygame.display.update()         #update everything to screen
    clock.tick(120)                  #set the framerate - while loop will not run faster than 60 times per second





