import json
import os

'''
>> picture_position
0: abstain
1: front
2: back
3: driver_side
4: assistant_side
5. inside

>> damage
0: non-severe damage
1: severe damage

>> damage_confidence
0: abstain
1: exact damage

>> objects
list of polygons


>> objects
{
    'type': 'polygon',
    'points': [
        []
    ],
    'class_name':'2' (str),
    'class_confidence':'1' 
}
    >> points (polygon)

    >> class_name
    1: dent
    2: scratch

    >> class_confidence
    0: unable to recognize dent/scratch/dirt/dust/light distortion/
    1: accurate dent/scratch

'''


label_dir_path = '/Users/kp/Desktop/work/scratch_detection/socar_dataset/bbox_labels/' #path of the label


label_file_names = os.listdir(label_dir_path)

label_file = label_file_names[0]

print(label_file)
print(label_dir_path + '/' + label_file)
with open(label_dir_path + '/' + label_file, 'r') as f:
    label_dict = json.load(f)

print(label_dict)
print(label_dict.keys())
print('\n')
print(label_dict['objects'])