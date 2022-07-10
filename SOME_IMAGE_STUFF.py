from PIL import Image

im = Image.open("c://users/kais/Desktop/sth.jpg")
mm = im.load()
x = 97
y = 477
print(mm[x,y])

