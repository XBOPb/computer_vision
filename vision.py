import cv2
import numpy as np


def find_objects(image):
    height, width, depth = image.shape
    image_blob = cv2.dnn.blobFromImage(image, 1/255, (608, 608), (0, 0, 0), swapRB=True, crop=False)
    network.setInput(image_blob)
    outs = network.forward(out_layers)
    class_indexes, class_scores, boxes = ([] for i in range(3))
    object_count = 0
    for out in outs:
        for object in out:
            scores = object[5:]
            class_index = np.argmax(scores)
            class_score = scores[class_index]
            if class_score > 0:
                center_x = int(object[0] * width)
                center_y = int(object[1] * height)
                object_width = int(object[2] * width)
                object_height = int(object[3] * height)
                box = [center_x - object_width // 2, center_y - object_height // 2, object_width, object_height]
                boxes.append(box)
                class_indexes.append(class_index)
                class_scores.append(float(class_score))

    chosen_boxes = cv2.dnn.NMSBoxes(boxes, class_scores, 0.0, 0.4)
    for box_index in chosen_boxes:
        box = boxes[box_index]
        class_index = class_indexes[box_index]

        if classes[class_index] in classes_to_look_for:
            object_count += 1
            image_to_process = draw_object_box(image, class_index, box)

    final_image = draw_object_count(image_to_process, object_count)
    return final_image


def draw_object_box(image, index, box):
    x, y, w, h = box
    start = (x, y)
    end = (x + w, y + h)
    color = (0, 255, 0)
    width = 2
    final_image = cv2.rectangle(image, start, end, color, width)

    start = (x, y - 10)
    font_size = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 2
    text = classes[index]
    final_image = cv2.putText(final_image, text, start, font, font_size, color, width, cv2.LINE_AA)
    return final_image


def draw_object_count(image, object_count):
    start = (45, 150)
    font_size = 2
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    width = 3
    text = "Objects found: " + str(object_count)
    font_color = (0, 0, 0)
    final_image = cv2.putText(image, text, start, font, font_size, font_color, width, cv2.LINE_AA)
    return final_image


def start_object_detection():
    try:
        image = cv2.imread('trucks.jpg')
        image = find_objects(image)

        cv2.imshow('Image', image)
        if cv2.waitKey(0):
            cv2.destroyAllWindows()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    network = cv2.dnn.readNetFromDarknet('yolov4-tiny.cfg', 'yolov4-tiny.weights')
    layer_names = network.getLayerNames()
    out_layers_indexes = network.getUnconnectedOutLayers()
    out_layers = [layer_names[index - 1] for index in out_layers_indexes]
    with open('coco.names') as file:
        classes = file.read().split("\n")

    classes_to_look_for = ['car']
    start_object_detection()