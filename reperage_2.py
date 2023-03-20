import os
import xml.etree.ElementTree as ET
from PIL import Image

data = 'data'
label = 'labels'
output = 'classification'

os.makedirs(output, exist_ok=True)

for xml in os.listdir(label):
    if xml.endswith('.xml'):
        xml_path = os.path.join(label, xml)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        image_filename = root.find('filename').text
        image_path = os.path.join(data, image_filename)

        image = Image.open(image_path)

        for obj in root.findall('object'):
            
            car_type = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            car_box = image.crop((xmin, ymin, xmax, ymax))

            car_output_dir = os.path.join(output, car_type)
            os.makedirs(car_output_dir, exist_ok=True)
            car_output_path = os.path.join(car_output_dir, f'{os.path.splitext(image_filename)[0]}_{car_type}.jpg')
            car_box.save(car_output_path)