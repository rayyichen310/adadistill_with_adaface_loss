import torch
import sys
import os
import argparse

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add the script directory to sys.path to allow importing backbones
sys.path.append(script_dir)

from backbones.mobilefacenet import MobileFaceNet

def convert_to_onnx(model_path, output_path, embedding_size=512):
    print(f"Loading model from {model_path}...")
    
    # Initialize the model
    # Assuming input size is (112, 112) which is standard for MobileFaceNet in this context
    model = MobileFaceNet(input_size=(112, 112), embedding_size=embedding_size)
    
    # Load the weights
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        # Handle case where state_dict might be inside a key like 'state_dict' or 'model'
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
            
        # Remove 'module.' prefix if it exists (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 112, 112)
    
    print(f"Exporting to ONNX at {output_path}...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print("Conversion complete!")
    except Exception as e:
        print(f"Error exporting to ONNX: {e}")

if __name__ == "__main__":
    # Define default paths relative to the script location
    default_model_path = os.path.join(script_dir, "output/AdaDistill/adaptive_geo_backbone.pth")
    default_output_path = os.path.join(script_dir, "output/AdaDistill/adaptive_geo_backbone.onnx")

    parser = argparse.ArgumentParser(description="Convert MobileFaceNet .pth model to ONNX")
    parser.add_argument("--model_path", type=str, default=default_model_path, help="Path to the .pth model file")
    parser.add_argument("--output_path", type=str, default=default_output_path, help="Path to save the .onnx model file")
    parser.add_argument("--embedding_size", type=int, default=512, help="Embedding size of the model")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        sys.exit(1)
        
    convert_to_onnx(args.model_path, args.output_path, args.embedding_size)
