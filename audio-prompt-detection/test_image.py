from src.image_utils import predict_image

 # The 'r' at the start is the magic fix!
image_path = r"E:\new_hack_iitk\audio-prompt-detection\fake_10015.jpg"

label, confidence = predict_image(image_path)

print("\nFinal Output:")
print("Label:", label)
print("Confidence:", confidence)