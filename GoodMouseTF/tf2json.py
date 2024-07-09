import flatbuffers
import tflite.Model

# Load TFLite model
with open('./GoodMouse/models/mediapipe/hand_detector.tflite', 'rb') as f:
    buf = f.read()

# Get the root object in the FlatBuffer
model = tflite.Model.GetRootAsModel(buf, 0)

# Convert to JSON
import json

def model_to_dict(model):
    model_dict = {}
    model_dict['version'] = model.Version()
    model_dict['subgraphs'] = []

    for i in range(model.SubgraphsLength()):
        subgraph = model.Subgraphs(i)
        subgraph_dict = {}
        subgraph_dict['tensors'] = []
        for j in range(subgraph.TensorsLength()):
            tensor = subgraph.Tensors(j)
            tensor_dict = {
                'name': tensor.Name().decode('utf-8'),
                'shape': tensor.ShapeAsNumpy().tolist(),
                'type': tensor.Type()
            }
            subgraph_dict['tensors'].append(tensor_dict)
        model_dict['subgraphs'].append(subgraph_dict)
    return model_dict

model_dict = model_to_dict(model)

# Save to JSON file
with open('model.json', 'w') as json_file:
    json.dump(model_dict, json_file, indent=4)
