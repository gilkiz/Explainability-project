from explainability.utils import load_model
import sys

model_dir = r"/home/noamkeidar/beatbox-research/models"
model_filename = "Exp2122_ECG_All_in_One_GOLD_GPU0_Right_Bundle_Branch_Block_epoch_49.pt"
sys.path.append("/home/noamkeidar/beatbox-research/src/beatbox_research")


model = load_model(path=model_dir, model_name=model_filename, device="cuda:0")

print("Success!!")
