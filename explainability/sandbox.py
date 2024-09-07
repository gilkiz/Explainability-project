from explainability.utils import load_model

model_dir = r"C:\Users\noam\OneDrive\Desktop\BeatBoxAI\models"
model_filename = "Exp2122_ECG_All_in_One_GOLD_GPU0_Right_Bundle_Branch_Block_epoch_49.pt"


model = load_model(path=model_dir, model_name=model_filename, device="cuda:0")

print("Success!!")
