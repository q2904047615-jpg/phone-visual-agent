"""Add an actual Android camera icon sample, without operating any phone."""
import argparse
from pathlib import Path
import re
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--box',type=int,nargs=4,required=True,metavar=('LEFT','TOP','RIGHT','BOTTOM'))
    parser.add_argument('--name',default='android_bag')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.name):
        parser.error('模板名称只允许英文、数字、横线或下划线。')
    with Image.open(args.image) as image:
        left,top,right,bottom = args.box
        if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
            parser.error('裁剪框必须在图片内且非空。')
        directory = Path(__file__).resolve().parent/'poc/features/lucky_bag/templates'
        directory.mkdir(exist_ok=True)
        target = directory/(args.name+'.png')
        if target.exists():
            parser.error('同名模板已存在，请换一个名称。')
        image.crop(args.box).convert('RGB').save(target)
        print('已追加模板：'+str(target))

if __name__ == '__main__':
    main()
