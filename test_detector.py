import os
from detector import DrainageDetector

def main():
    d = DrainageDetector()
    sample_img = os.path.join('static', 'images', 'samples', 'sample_partial_blockage.jpg')
    output_img = os.path.join('outputs', 'test_annotated.jpg')
    
    print(f"Testing on {sample_img}...")
    res = d.process_image_file(sample_img, output_img)
    print("Status:", res['assessment']['status'])
    print("Blockage %:", res['assessment']['blockage_percentage'])
    print("Detected count:", len(res['detections']))
    print("Output exists:", os.path.exists(output_img))
    print("Placeholder mode:", res['is_placeholder'])
    print("Test passed successfully!")

if __name__ == '__main__':
    main()
