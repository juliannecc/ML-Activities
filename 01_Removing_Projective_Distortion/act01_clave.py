from tkinter import *
from tkinter import filedialog
import os
from PIL import Image, ImageTk
import numpy as np

src_coords = []

# gets the filepath of the image
def browse_img():
    global filepath, src_w, src_h
    file = filedialog.askopenfile(mode='r', filetypes=[('PNG', '*.png'), ('JPG', '*.jpg'), ('JPEG', '*.jpeg')])
    if file:
        filepath = os.path.abspath(file.name)
        src_w,src_h = get_image_size(filepath)
        button_import.config(state="disabled")
        get_coordinates()

# gets the image dimensions and resizes accordingly
def get_image_size(fp):
    img = Image.open(fp)
    w,h = img.size
    # resize_scale = 100
    if w > 640 or h > 720:
        for scale in range(95,40,-5):
            w,h = (int(scale*w/100), int(scale*h/100))
            if w < 640 and h < 720:
                # resize_scale = scale
                break
    return w,h

# imports to tkinter canvas for coordinate marking
def get_coordinates():
    pilImage = Image.open(filepath)
    pilImage = pilImage.resize((src_w,src_h))
    src_img = ImageTk.PhotoImage(pilImage)
    src_img_canvas = src_canvas.create_image(0,0,image = src_img, anchor=NW)
    src_canvas.bind('<Button-1>', mark_coordinates)
    root.mainloop()

# marks coordinates on tkinter canvas
def mark_coordinates(event):    
    if len(src_coords) < 4:
        src_coords.append([event.x,event.y])
        src_canvas.create_oval(event.x - 7, event.y - 7, event.x + 7, event.y+ 7, fill="blue")
    if len(src_coords) == 4:
        button_correct.config(state="normal")

# corrects the order: ul, ur, ll, lr
def get_order (coordinates): 
    coordinates.sort(key = lambda x: x[1])
    upper = [coordinates[0],coordinates[1]]
    upper.sort()
    lower = [coordinates[2],coordinates[3]]
    lower.sort()
    return [upper[0],upper[1], lower[0],lower[1]]

# gets the matrix required
def homography():
    # https://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d-2d-planar-homog
    # https://math.stackexchange.com/questions/296794/finding-the-transform-matrix-from-4-projected-points-with-javascript
    ordered_src_coords = get_order(src_coords)
    dest_coords = [[0,0],[640,0],[0,720],[640,720]]
    
    p_i = np.zeros((8,9))
    for i in range(4):
        src_x, src_y = ordered_src_coords[i]
        dest_x, dest_y = dest_coords[i]
        p_i[i*2,:] = [src_x, src_y, 1, 0, 0, 0, -dest_x*src_x, -dest_x*src_y, -dest_x]
        p_i[i*2+1, :] = [0, 0, 0, src_x, src_y, 1, -dest_y*src_x, -dest_y*src_y, -dest_y]
    [U,S,V]=np.linalg.svd(p_i)
    m = V[-1,:] / V[-1,-1]
    H = np.reshape(m,(3,3))
    return H

# orders the matrix accordingly [x-> y, y-> x]
def to_mat(mat):
    x,y,z = mat.shape
    new_mat = np.zeros((y,x,z))
    for i in range(x): new_mat[:,i] = mat[i]
    return new_mat.astype(int)

# switches back the order of matrix to the original [y-> x, x-> y]
def to_original(mat):
    x,y,z = mat.shape
    img = np.zeros((y,x,z))
    for i in range(x): img[:,i] = mat[i]
    return img.astype(int)

# gets the resulting image
def correct():
    # https://stackoverflow.com/questions/2659312/how-do-i-convert-a-numpy-array-to-and-display-an-image
    # https://stackoverflow.com/questions/52154943/how-to-interpret-the-pil-image-in-numpy-array
    # https://stackoverflow.com/questions/66388578/python-using-image-pil-to-show-an-image-returns-a-black-square
    H = homography()
    pilImage = Image.open(filepath)
    pilImage = pilImage.resize((src_w,src_h)).convert("RGB")
    
    pilMat = to_mat(np.array(pilImage))
    corrected_img = np.zeros((640,720,3))

    for i in range(corrected_img.shape[0]):
        for j in range(corrected_img.shape[1]):
            mat = np.dot(H, [i,j,1])
            k,l,_ = (mat / mat[2]).astype(int)
            if (k >= 0 and k < 640) and (l >= 0 and l < 720):
                    corrected_img[k,l] = pilMat[i,j]
    corrected_img = to_original(corrected_img)

    pilArray = Image.fromarray(corrected_img.astype('uint8'),'RGB')
    pilCorrected = ImageTk.PhotoImage(pilArray)
    corrected_img_canvas = dest_canvas.create_image(0,0,image = pilCorrected, anchor=NW)
    root.mainloop()

if __name__ == "__main__":
    root = Tk()
    root.title("Projective Distortion Remover")
    root.geometry("1280x770")

    button_import = Button(root, text="     Import Image     ", font=("Helvetica,20"), command=browse_img)
    button_correct = Button(root, text="     Correct Distortion     ", font=("Helvetica,20"),state="disabled", command=correct)
    button_import.grid(row=0, column=0,pady=2)
    button_correct.grid(row=0,column=1,pady=2)

    src_canvas = Canvas(height=720,width=640,bg="white")
    dest_canvas = Canvas(height=720, width=640, bg="gray")
    src_canvas.grid(row=2,column=0)
    dest_canvas.grid(row=2,column=1)

    root.mainloop()