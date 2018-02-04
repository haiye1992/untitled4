from PIL import Image
from PIL import ImageFilter


print("hell")
x = [1, 2, 3, 4]
val = [x*x for x in x]
print(val)
word = Image.open('D:/IMG_0002.JPG')
assert isinstance(word,Image.Image)
print(word.format, word.size, word.mode)
#word.rotate(90).show()
size=(1280,1280)
word.thumbnail(size)
outfile='D:/1.JPG'
word.save(outfile)
word.transpose(Image.FLIP_TOP_BOTTOM).show()
word.convert("L").show()
word1=word.filter(ImageFilter.BLUR)
word1.show()



