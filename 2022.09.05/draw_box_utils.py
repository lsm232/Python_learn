import torch
import PIL.ImageFont as ImageFont
from PIL import ImageDraw,ImageColor
from PIL import Image
import numpy as np
from torchvision.transforms import functional as F

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

def draw_box_and_mask(
        image:Image,
        boxes:np.ndarray,
        mask:np.ndarray,

        scores:np.ndarray,
        box_thresh:float,
        mask_thresh:float,
        classes:np.ndarray,
        font:str='arial.ttf',
        font_size:int=24,
        thickness: int=8,
        draw_box_on_image=True,
        draw_mask_on_image=False,
        category_index:dict={},
):
    if draw_box_on_image:
        #过滤掉一些box
        idx=np.greater(scores,box_thresh)
        used_boxes=boxes[idx]
        used_classes=classes[idx]
        used_scores=scores[idx]
        colors=[STANDARD_COLORS[int(cls)%len(STANDARD_COLORS)] for cls in classes]

        d_image=ImageDraw.Draw(image)
        for box,cls,score,color in zip(used_boxes,used_classes,used_scores,colors):
            left,top,right,bottom=box
            d_image.line([(left,top),(right,top),(right,bottom),(left,bottom),(left,top)],fill=color,width=thickness)
            draw_text(d_image,box.tolist(),category_index,score,str(int(cls)),color=color)
        return image

def draw_text(
        draw,
        box:list,
        classes_dict:dict,
        score:float,
        cls:str,
        font:str = 'arial.ttf',
        font_size:int=12,
        color:str='sdadad',

):
    font=ImageFont.truetype(font,font_size)
    #文字
    display_str=f"{classes_dict[cls]}:{score*100}%"
    max_height=max([font.getsize(ds)[1] for ds in display_str])

    #显示分类信息的位置
    left,top,right,bottom=box
    if top>max_height:
        text_top=top-max_height*(1+0.05*2)
        text_bottom=top
    else:
        text_top=bottom
        text_bottom=bottom+max_height

    #一个字一个字的画框
    for ds in display_str:
        text_width, text_height = font.getsize(ds)
        margin=np.ceil(0.05 * text_width)
        draw.rectangle([(left,text_top),(left+2*margin+text_width,text_bottom)],fill=color)
        draw.text((left+margin,text_top),text=ds,fill='black',font=font)
        left+=text_width




