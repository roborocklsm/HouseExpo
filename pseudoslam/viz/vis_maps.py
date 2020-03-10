import os
import json
import cv2
import argparse
import numpy as np
from matplotlib import pyplot as plt

from PIL import ImageDraw
from PIL import Image

meter2pixel = 20
border_pad = 20*3
ALLOWED_ROOM_TYPES = {'indoor':3, 'kitchen':4, 'dining_room':5, 'living_room':6,
                      'bathroom':7, 'bedroom':8, 'office':9, 'hallway':10}
beds_meter = [[2.0,2.0],[1.8, 2.0], [1.5, 2.0], [1.2 ,2.0]]
beds_pixels = [[int(bed_meter[0]*meter2pixel), int(bed_meter[1]*meter2pixel)] for bed_meter in beds_meter]

barriers_meter = [[0.5, 0.5],[0.4, 0.4],[0.3 ,0.3],[0.5, 0.4],[0.5,0.3],[0.4,0.3]]
# barriers_meter = [[0.3, 0.3],[0.2, 0.2],[0.1 ,0.1],[0.3, 0.2],[0.3,0.1],[0.2,0.1]]
barriers_pixels = [[int(barrier_meter[0]*meter2pixel), int(barrier_meter[1]*meter2pixel)] for barrier_meter in barriers_meter]

height, width = 512, 512

bx1 = int(0.95*width)
by1 = int(0.95*height)
bx0 = int(0.05*width)
by0 = int(0.05*height)

def get_cloest_pt(x,y):
    pt0 = np.array((x,y))
    pts = [np.array((x, by0)), np.array((x, by1)), np.array((bx0, y)), np.array((bx1, y))]
    dis = [np.linalg.norm(pt0-pt) for pt in pts]
#     print(np.amin(dis))
    return pts[np.argmin(dis)]

def check_closest_pt(x, y, bx, by, ref_image):
    cnt = 0
    if x == bx:
        for _y in range(min(y, by), max(y,by)):
            if ref_image[x, _y] != 0:
                cnt += 1;
    elif y == by:
        for _x in range(min(x, bx), max(x,bx)):
            if ref_image[_x, y] != 0:
                cnt += 1;
    if cnt > 5: 
        return False
    return True

def darw_poly(draw, height, width, fil_col, ref_image, coinf=50, draw_pts=True):
    x_array, y_array=np.where(ref_image==2)
    px0,py0 = 0,0
    cpx, cpy = 511, 511
    while ref_image[px0,py0] != 1 or check_closest_pt(px0, py0, cpx, cpy, ref_image):
        idx = np.random.randint(x_array.shape[0])
        px0 = x_array[idx]
        py0 = y_array[idx]
        px0 = px0+np.random.randint(20)-10
        py0 = py0+np.random.randint(20)-10
        
        cpx, cpy = get_cloest_pt(px0, py0)
    
    rand_num = np.random.rand()
    px1, py1 = int(coinf*(np.random.rand()-0.5) + cpx), int(-coinf*(np.random.rand()-0.5) + cpy)
    px2, py2 = int(coinf*(np.random.rand()-0.5) + cpx), int(-coinf*(np.random.rand()-0.5) + cpy)

    for i in range(np.random.randint(1,40)):
        rand_x = np.random.rand()
        rand_y = np.random.rand()
        x, y = rand_x*px1+(1-rand_x)*px2, rand_y*py1+(1-rand_y)*py2
        draw.line((px0, py0, x, y), fil_col)
        
        if draw_pts:
            draw.point([(x, y)], 2)
            dot_enhance_num = np.random.randint(2,15)
            for _ in range(dot_enhance_num):
                add_x, add_y = np.random.rand(2)-0.5
                if rand_num < 0.5:
                    add_x, add_y = int(add_x*2), int(add_y*3)
                else:
                    add_x, add_y = int(add_x*3), int(add_y*2)
                draw.point([(x+add_x, y+add_y)], 2)
    return px0, py0, px1, py1, px2, py2

def _get_room_tp_id(room):
    room = room.lower()
    if room == 'toilet':
        room = 'bathroom'
    # elif room == 'guest_room':
    #     room = 'bedroom'
    if room not in ALLOWED_ROOM_TYPES:
        return ALLOWED_ROOM_TYPES['indoor']
    return ALLOWED_ROOM_TYPES[room]

def draw_map(file_name, json_path, save_path):
    print("Processing ", file_name)

    with open(json_path + '/' + file_name + '.json') as json_file:
        json_data = json.load(json_file)

    # Draw the contour
    verts = (np.array(json_data['verts']) * meter2pixel).astype(np.int)
    x_max, x_min, y_max, y_min = np.max(verts[:, 0]), np.min(verts[:, 0]), np.max(verts[:, 1]), np.min(verts[:, 1])
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    ######################################
    raw_vis_map = np.ones((y_max - y_min + border_pad * 2,
                        x_max - x_min + border_pad * 2))*127
    if raw_vis_map.shape[0]>512 or raw_vis_map.shape[1]>512:
        return

    verts[:, 0] = verts[:, 0] - x_min + border_pad
    verts[:, 1] = verts[:, 1] - y_min + border_pad
    cv2.drawContours(raw_vis_map, [verts], -1, 255, -1)
    cv2.drawContours(raw_vis_map, [verts], -1, 0, 1)
    # Save map
    if not os.path.exists(save_path + "/raw_vis"):
        os.mkdir(save_path + "/raw_vis")
    cv2.imwrite(save_path + "/raw_vis/" + file_name + '.png', raw_vis_map)
    ######################################
    raw_map = np.zeros((y_max - y_min + border_pad * 2,
                        x_max - x_min + border_pad * 2))
    cv2.drawContours(raw_map, [verts], -1, 1, -1)
    cv2.drawContours(raw_map, [verts], -1, 2, 1)
    # Save map
    if not os.path.exists(save_path + "/raw"):
        os.mkdir(save_path + "/raw")
    cv2.imwrite(save_path + "/raw/" + file_name + '.png', raw_map)
    ######################################
    seg_map = np.zeros((y_max - y_min + border_pad * 2,
                        x_max - x_min + border_pad * 2))

    cv2.drawContours(seg_map, [verts], -1, 1, -1)
    cv2.drawContours(seg_map, [verts], -1, 2, 1)

    for room_cate in json_data['room_category']:
        room_id = _get_room_tp_id(room_cate)
        for box in json_data['room_category'][room_cate]:
            x0 = int(box[0]*meter2pixel)+border_pad-x_min
            y0 = int(box[1]*meter2pixel)+border_pad-y_min
            x1 = int(box[2]*meter2pixel)+border_pad-x_min
            y1 = int(box[3]*meter2pixel)+border_pad-y_min
            tmp_map = np.zeros(seg_map.shape)
            tmp_map[y0:y1, x0:x1] = 1
            seg_map[np.where((seg_map!=0)*(seg_map!=2)*(tmp_map==1))] = room_id

    # Save map
    if not os.path.exists(save_path + "/seg"):
        os.mkdir(save_path + "/seg")
    cv2.imwrite(save_path + "/seg/" + file_name + '.png', seg_map)
    ######################################
    inp_map = np.zeros((y_max - y_min + border_pad * 2,
                        x_max - x_min + border_pad * 2)).astype(np.uint8)

    cv2.drawContours(inp_map, [verts], -1, 1, -1)
    cv2.drawContours(inp_map, [verts], -1, 2, 1)


    # for room_cate in json_data['room_category']:
    #     room_id = _get_room_tp_id(room_cate)
    #     if room_id not in [4,5,6,9,8]:
    #         for box in json_data['room_category'][room_cate]:
    #             x0 = int(box[0]*meter2pixel)+border_pad-x_min
    #             y0 = int(box[1]*meter2pixel)+border_pad-y_min
    #             x1 = int(box[2]*meter2pixel)+border_pad-x_min
    #             y1 = int(box[3]*meter2pixel)+border_pad-y_min
    #             tmp_map = np.zeros(inp_map.shape)
    #             tmp_map[y0:y1, x0:x1] = 1
    #             inp_map[np.where((inp_map!=0)*(inp_map!=2)*(tmp_map==1))] = room_id

    for room_cate in json_data['room_category']:
        room_id = _get_room_tp_id(room_cate)
        # if room_id in [4,5,6,9,8]:
            # add 
        if room_id == 8:
            idx = np.random.randint(len(beds_pixels))
            scale_pixels = beds_pixels
        else:
            idx = np.random.randint(len(barriers_pixels))
            scale_pixels = barriers_pixels
        for box in json_data['room_category'][room_cate]:
            x0 = int(box[0]*meter2pixel)+border_pad-x_min
            y0 = int(box[1]*meter2pixel)+border_pad-y_min
            x1 = int(box[2]*meter2pixel)+border_pad-x_min
            y1 = int(box[3]*meter2pixel)+border_pad-y_min
            try:
                tmp_map = np.zeros(inp_map.shape)
                tmp_map[y0:y1, x0:x1] = 1
                y_array, x_array = np.where((inp_map!=0)*(tmp_map==1))
                x00_lim = x_array.min()
                x11_lim = x_array.max()
                y00_lim = y_array.min()
                y11_lim = y_array.max()

                if np.random.randn()<0:
                    # bed ver
                    bed_w = scale_pixels[idx][0]
                    bed_h = scale_pixels[idx][1]
                else:
                    # bed hor
                    bed_w = scale_pixels[idx][1]
                    bed_h = scale_pixels[idx][0]
                # case = np.random.randint(4)
                add_bed_flag = False
                if not add_bed_flag:
                    x00 = x00_lim+np.random.randint(x11_lim-x00_lim-bed_h)
                    y00 = y00_lim
                    x11 = x00+bed_h
                    y11 = y00+bed_w
                    if (inp_map[y00:y11, x00:x11] != 0).all():
                        inp_map[y00-1:y11+1, x00-1:x11+1] = 19
                        add_bed_flag = True
                if not add_bed_flag:
                    x00 = x00_lim
                    y00 = y00_lim+np.random.randint(y11_lim-y00_lim-bed_w)
                    x11 = x00+bed_h
                    y11 = y00+bed_w
                    if ((inp_map[y00:y11, x00:x11] != 0) * \
                    (inp_map[y00:y11, x00:x11] != 2)).all():
                        inp_map[y00-1:y11+1, x00-1:x11+1] = 19
                        add_bed_flag = True
                if not add_bed_flag:
                    x11 = x11_lim-np.random.randint(x11_lim-x00_lim-bed_h)
                    y11 = y11_lim
                    x00 = x11-bed_h
                    y00 = y11-bed_w
                    if (inp_map[y00:y11, x00:x11] != 0).all():
                        inp_map[y00-1:y11+1, x00-1:x11+1] = 19
                        add_bed_flag = True
                if not add_bed_flag:
                    x11 = x11_lim
                    y11 = y11_lim-np.random.randint(y11_lim-y00_lim-bed_w)
                    x00 = x11-bed_h
                    y00 = y11-bed_w
                    if (inp_map[y00:y11, x00:x11] != 0).all():
                        inp_map[y00-1:y11+1, x00-1:x11+1] = 19
                        add_bed_flag = True
            
                if add_bed_flag:
                    if room_id == 8:
                        print("=============== add bed: {}".format(file_name))
                    else:
                        print("=============== add barry: {}".format(file_name))
                    # print(x00, y00, x11, y11)
                    # print(case)
                    # print(bed_h, bed_w)
                    # print(x0, y0, x1, y1)
            except:
                pass

            tmp_map = np.zeros(inp_map.shape)
            tmp_map[y0:y1, x0:x1] = 1
            inp_map[np.where((inp_map!=19)*(inp_map!=0)*(inp_map!=2)*(tmp_map==1))] = room_id
            inp_map[np.where(inp_map==19)] = 0
    
    inp_map = (inp_map != 0).astype(np.uint8)
    contours, hierarchy = cv2.findContours(inp_map, 2, 1)
    cv2.drawContours(inp_map, contours, -1, 2)
    x_boundary, y_boundary = np.where(inp_map == 2)
    abandoned_boundary_idx = np.random.random(x_boundary.shape)<0.1
    x_boundary = x_boundary[abandoned_boundary_idx]
    y_boundary = y_boundary[abandoned_boundary_idx]
    inp_map[(x_boundary,y_boundary)] = 0
    x_boundary, y_boundary = np.where(inp_map == 2)
    abandoned_boundary_idx = np.random.random(x_boundary.shape)<0.1
    x_boundary = x_boundary[abandoned_boundary_idx]
    y_boundary = y_boundary[abandoned_boundary_idx]
    inp_map[(x_boundary,y_boundary)] = 1

    inp = np.zeros((height, width)).astype(np.uint8)
    inp[(height-inp_map.shape[0])//2:(height-inp_map.shape[0])//2+inp_map.shape[0],\
         (width-inp_map.shape[1])//2:(width-inp_map.shape[1])//2+inp_map.shape[1]] = inp_map

    tar = np.zeros((height, width)).astype(np.uint8)
    tar[(height-raw_map.shape[0])//2:(height-raw_map.shape[0])//2+raw_map.shape[0],\
         (width-raw_map.shape[1])//2:(width-raw_map.shape[1])//2+raw_map.shape[1]] = raw_map

    # freespace enhance
    dot_num = 100
    cnt = 0
    cnt2 = 0

    bx1 = int(0.95*width)
    by1 = int(0.95*height)
    bx0 = int(0.05*width)
    by0 = int(0.05*height)

    for _ in range(2500):
        randcoinf = np.random.random()
        if randcoinf<0.01:
            x,y = np.random.rand(2)
            x = int(x*height)
            y = int(y*width)        
            w = np.random.randint(2,10)
            h = np.random.randint(2,10)
            x = max(min(x, bx1), bx0)
            y = max(min(y, by1), by0)
            # if np.unique(inp[x-h//2:x+h//2, y-h//2:y+h//2]).shape[0]>1:
            if inp[x,y] in [1,2]:
                inp[x-w//2:x+w//2,y-h//2:y+h//2] = np.zeros((w//2*2,h//2*2))
        elif randcoinf<0.1:
            x,y = np.random.rand(2)
            x = int(x*height)
            y = int(y*width)        
            w = np.random.randint(2,5)
            h = np.random.randint(2,5)
            x = max(min(x, bx1), bx0)
            y = max(min(y, by1), by0)
            # if np.unique(inp[x-h//2:x+h//2, y-h//2:y+h//2]).shape[0]>1:
            if inp[x,y] in [1,2]:
                inp[x-w//2:x+w//2,y-h//2:y+h//2] = np.ones((w//2*2,h//2*2))*2
        else:
            x,y = np.random.rand(2)
            x = int(x*height)
            y = int(y*width)        
            w = np.random.randint(2,5)
            h = np.random.randint(2,5)
            x = max(min(x, bx1), bx0)
            y = max(min(y, by1), by0)
            if np.unique(inp[x-h//2:x+h//2, y-h//2:y+h//2]).shape[0]>1:
            # if inp[x,y] in [1,2]:
                inp[x-w//2:x+w//2,y-h//2:y+h//2] = np.ones((w//2*2,h//2*2))*2


    for _ in range(dot_num):
        x,y = np.random.rand(2)
        x = int(x*height)
        y = int(y*width)
        if inp[x,y] == 2:
            w = np.random.randint(2,5)
            h = np.random.randint(2,5)
            random_num = np.random.rand()
            if random_num<0.8:
                inp[x-w//2:x+w//2,y-h//2:y+h//2] = np.ones((width,height))[x-w//2:x+w//2,y-h//2:y+h//2]*2
            elif random_num<0.9:
                inp[x-w*2:x+w*2,y-h*2:y+h*2] = np.ones((width,height))[x-w*2:x+w*2,y-h*2:y+h*2]*1
            else:
                inp[x-w*2:x+w*2,y-h*2:y+h*2] = np.zeros((width,height))[x-w*2:x+w*2,y-h*2:y+h*2]

        if inp[x,y] == 1:
            if cnt <20:
                if cnt2 < 10:
                    cnt2 += 1
                    w = np.random.randint(1,5)
                    h = np.random.randint(1,5)
                    inp[x-w//2:x+w//2,y-h//2:y+h//2] = np.zeros((width,height))[x-w//2:x+w//2,y-h//2:y+h//2]
                else:
                    cnt += 1
                    w = np.random.randint(1,3)
                    h = np.random.randint(1,3)
                    inp[x-w//2:x+w//2,y-h//2:y+h//2] = np.ones((width,height))[x-w//2:x+w//2,y-h//2:y+h//2]*2
    #             else:
    #                 inp[x,y] = 0
    image =  Image.fromarray(inp.copy())
    draw = ImageDraw.Draw(image)  

    #     绘制飞线
    rect_num = 10
    for i in range(rect_num):
        darw_poly(draw, height, width, 1, inp, 50)
    inp = np.array(image) 

    inp_map = np.concatenate([tar, inp], 1)

    inp_map[np.where(inp_map==0)] = 127
    inp_map[np.where(inp_map==1)] = 255
    inp_map[np.where(inp_map==2)] = 0

    # Save map
    if not os.path.exists(save_path + "/inp"):
        os.mkdir(save_path + "/inp")
    cv2.imwrite(save_path + "/inp/" + file_name + '.png', inp_map)
    #####################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize the subset of maps in .png.")
    parser.add_argument("map_id_set_file", help="map id set (.txt)")
    parser.add_argument("--json_path", type=str, default="./HouseExpo/json/", help="json file path")
    parser.add_argument("--save_path", type=str, default='./png')
    result = parser.parse_args()

    json_path = os.path.abspath(os.path.join(os.getcwd(), result.json_path))
    map_file = os.path.abspath(os.path.join(os.getcwd(), result.map_id_set_file))
    save_path = os.path.abspath(os.path.join(os.getcwd(), result.save_path))
    print("---------------------------------------------------------------------")
    print("|map id set file path        |{}".format(map_file))
    print("---------------------------------------------------------------------")
    print("|json file path              |{}".format(json_path))
    print("---------------------------------------------------------------------")
    print("|Save path                   | {}".format(save_path))
    print("---------------------------------------------------------------------")

    map_ids = np.loadtxt(map_file, str)

    for map_id in map_ids:
        draw_map(map_id, json_path, save_path)

    print("Successfully draw the maps into {}.".format(save_path))
