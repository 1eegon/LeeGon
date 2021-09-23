from PIL import Image
import os
import glob

files = glob.glob('./image/image/*')

count = 0

for f in files:
    try:
        img = Image.open(f)
        img_resize = img.resize((300, 300))
        title, ext = os.path.splitext(f)
        count += 1
        img_resize.save('./image/resizing/{0}.png'.format(count), 'PNG')

    except OSError as e:
        pass

