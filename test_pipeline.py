"""
Test upload of an image file and test processing a short synthetic video
"""
import os
import cv2
import numpy as np
from app import app
from detector import DrainageDetector

def test_full_pipeline():
    detector = DrainageDetector()
    
    # 1. Create a 3-second test video (15 frames) of drain
    video_path = os.path.join('uploads', 'test_synthetic_drain.mp4')
    output_video_path = os.path.join('outputs', 'annotated_test_synthetic.mp4')
    
    sample_img_path = os.path.join('static', 'images', 'samples', 'sample_partial_blockage.jpg')
    frame = cv2.imread(sample_img_path)
    h, w = frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, 5.0, (w, h))
    for i in range(15):
        # slight noise shift to simulate video movement
        noisy_frame = frame.copy()
        if i % 2 == 0:
            cv2.circle(noisy_frame, (100 + i*5, 100), 5, (255, 255, 255), -1)
        writer.write(noisy_frame)
    writer.release()
    print(f"[+] Created synthetic test video: {video_path}")
    
    # Process video
    video_res = detector.process_video_file(video_path, output_video_path, sample_rate=2)
    print(f"[+] Video processed successfully! Total frames: {video_res['total_frames']}, Peak blockage: {video_res['peak_blockage']}%, Status: {video_res['final_status']}")
    assert os.path.exists(output_video_path), "Output video was not created!"
    
    # 2. Test uploading image via Flask client
    client = app.test_client()
    with open(sample_img_path, 'rb') as f:
        res = client.post('/analyze-image', data={'image': f}, content_type='multipart/form-data', follow_redirects=True)
        assert res.status_code == 200
        assert b"Inspection Diagnosis Report" in res.data
        print("[+] Image upload via Flask test client succeeded!")

    print("[SUCCESS] All end-to-end tests completed successfully!")

if __name__ == '__main__':
    test_full_pipeline()
