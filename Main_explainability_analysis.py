import os
import torch
import numpy as np
import cv2
import pandas as pd
import shutil


class Explainability_AIO:
    def __init__(
        self,
        inference_models=None,
        to_use_signal_extraction=False,
        save_signal_extraction=False,
    ):
        self.save_signal_extraction = save_signal_extraction
        self.orignal_image = None
        self.jacobian_image = None
        self.signal_extraction = None
        self.explainability_img = None
        self.inference_models = inference_models
        self.models = []
        available_gpus = [
            torch.cuda.device(i) for i in range(torch.cuda.device_count())
        ]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.in_w = 1650
        self.in_h = 880
        self.in_channels = 3
        self.image = None
        self.signal_extraction = None
        self.attention_mask_threshold = 100
        self.mask_width = 4
        self.attention_mask = None
        self.TO_SAVE_ATTN = True
        self.last_inference = None
        self.last_attn_img = None
        self.last_attn_img_normalized = None
        self.last_fused_img = None
        self.alpha = 0.5
        if to_use_signal_extraction:
            self.gain = 5
        else:
            self.gain = 2
        self.image_name = None
        self.title = None
        self.to_use_signal_extraction = to_use_signal_extraction
        self.to_always_save = False
        self.threshold = 10

    def load_inference_models(self, path):
        # Iterate over all models in path
        for model_name in os.listdir(path):
            if model_name.endswith(".pt"):
                model_path = os.path.join(path, model_name)
                saved_state = torch.load(
                    model_path, map_location=torch.device(self.device)
                )
                # Split path to get the model name
                model = saved_state["full_model"]
                model.load_state_dict(saved_state["model_state"])
                model.name = model_name[33:-3]
                model.to(self.device)
                model.eval()
                self.models.append(model)

    def set_image(self, image, image_name):
        self.explainability_img = image
        self.image_name = image_name
        image = cv2.resize(image, self.shape)
        # blur = self.process_frame(image)
        self.image = image
        self.inference_input = image
        self.video_inference = False

    def normalize_inference_input(self):
        self.inference_input = np.array(self.inference_input)
        self.inference_input = torch.as_tensor(
            self.inference_input, dtype=torch.float32
        )
        if self.video_inference:
            self.inference_input = self.inference_input.permute(0, 3, 1, 2)
        else:
            self.inference_input = self.inference_input.permute(2, 0, 1)
            self.inference_input = self.inference_input.unsqueeze(0)
        self.inference_input = self.inference_input.to(self.device)

    def extract_signal(self):
        mask = self.image
        M = (
            (mask[:, :, 0] < self.attention_mask_threshold)
            * (mask[:, :, 1] < self.attention_mask_threshold)
            * (mask[:, :, 2] < self.attention_mask_threshold)
        )
        M = (M * 255).astype(np.uint8)
        # Save image
        signal_extraction_ = M.copy()
        self.signal_extraction = M.copy()
        for i in range(self.mask_width):
            self.signal_extraction[0 : self.in_h - i, 0 : self.in_w - i] = (
                self.signal_extraction[0 : self.in_h - i, 0 : self.in_w - i]
                + M[i : self.in_h, i : self.in_w]
            )
        # Set upper limit as 255
        self.signal_extraction[self.signal_extraction > 255] = 255
        self.attention_mask = self.signal_extraction
        self.signal_extraction = signal_extraction_
        # if self.TO_SAVE_ATTN:
        #     cv2.imwrite(os.path.join(os.getcwd(), 'Images','Extraction_image_cv.jpg'), self.signal_extraction)
        #     cv2.imwrite(os.path.join(os.getcwd(), 'Images','Attention_image_cv_attention_mask.jpg'), self.attention_mask)
        return M

    def get_data(self):
        return self.explainability_img

    def predict(self, relevant_models=None):
        self.normalize_inference_input()
        self.predictions = {}
        self.is_from_mobile = {}
        # Zero gradients
        for model in self.models:
            x_s = self.inference_input
            x_s = x_s.to(self.device).requires_grad_(False)
            if x_s.grad is not None:
                x_s.grad.zero_()
            x_s[:, 0, :, :] = (x_s[:, 0, :, :] - torch.min(x_s[:, 0, :, :])) / (
                torch.max(x_s[:, 0, :, :]) - torch.min(x_s[:, 0, :, :])
            )
            x_s[:, 1, :, :] = (x_s[:, 1, :, :] - torch.min(x_s[:, 1, :, :])) / (
                torch.max(x_s[:, 1, :, :]) - torch.min(x_s[:, 1, :, :])
            )
            x_s[:, 2, :, :] = (x_s[:, 2, :, :] - torch.min(x_s[:, 2, :, :])) / (
                torch.max(x_s[:, 2, :, :]) - torch.min(x_s[:, 2, :, :])
            )
            x_s = x_s.to(self.device).requires_grad_(True)
            model.zero_grad()
            model.eval()
            output = model(x_s)
            # self.explainability_img = self.jacobian_image * self.signal_extraction
            output_ae, output_lp, is_from_mobile = output
            # output_lp = torch.sigmoid(output_lp)
            # # Explainability
            output_lp.backward()
            J = x_s.grad
            Js = J.squeeze()
            # Take only positive values
            # Js[Js<0] = 0
            Js_abs = torch.abs(Js)
            # Js_abs = Js[0,:,:] * Js[1,:,:] * Js[2,:,:]
            # Js_abs = Js_abs.unsqueeze(0)
            # # Make color from grayscale
            # Js_abs = Js_abs.repeat(3,1,1)
            # Js_abs = torch.abs(Js)
            self.last_attn_img = Js_abs.permute(1, 2, 0).cpu().numpy()
            # Add median filter
            self.last_attn_img = cv2.medianBlur(self.last_attn_img, 3)
            self.last_attn_img_normalized = (
                self.last_attn_img / np.max(self.last_attn_img) * 255
            ).astype(np.uint8)
            threshold = self.threshold
            mask = (
                (self.last_attn_img_normalized[:, :, 0] > threshold)
                * (self.last_attn_img_normalized[:, :, 1] > threshold)
                * (self.last_attn_img_normalized[:, :, 2] > threshold)
            )
            # # Convert Binary mask to 3 channels
            mask = mask.astype(np.uint8)
            mask = mask * 255
            mask = mask[:, :, np.newaxis]
            mask = np.repeat(mask, 3, axis=2)
            # Make marker red
            mask[:, :, 2] = 0
            mask[:, :, 1] = 0
            # Plot mask
            # import matplotlib.pyplot as plt
            # plt.imshow(mask)
            # plt.show()

            # Draw mask on image
            self.last_attn_img_normalized = self.last_attn_img_normalized * (
                mask / 255
            ).astype(np.uint8)
            output_ae = output_ae.squeeze()
            output_ae = output_ae.detach().permute(1, 2, 0).cpu().numpy()
            output_ae = output_ae / np.max(output_ae)
            output_ae[output_ae < 0.0] = 0
            output_ae = output_ae * 255
            output_ae = output_ae.astype(np.uint8)
            # Save signal extraxtion image
            # if self.save_signal_extraction:
            #     cv2.imwrite(os.path.join(os.getcwd(),'Images',f'{self.title}_{model.name}_{self.image_name}_Signal_extraction.jpg'), output_ae)
            if self.to_use_signal_extraction:
                self.last_attn_img_normalized = (
                    self.last_attn_img_normalized * output_ae
                ).astype(np.uint8)
            original_image = (
                x_s.detach().squeeze().permute(1, 2, 0).cpu().numpy() * 255
            ).astype(np.uint8)
            beta = 1.0 - self.alpha
            self.last_fused_img = cv2.addWeighted(
                original_image,
                self.alpha,
                self.last_attn_img_normalized * self.gain,
                beta,
                0.0,
            )
            self.last_fused_img = increase_brightness(self.last_fused_img, value=100)
            is_from_mobile = torch.sigmoid(is_from_mobile)
            output_lp = output_lp.detach().cpu().numpy()
            is_from_mobile = is_from_mobile.detach().cpu().numpy()
            print(
                f"Prediction {model.name}: {np.mean(output_lp).item()}, is_from_mobile: {np.mean(is_from_mobile).item()}"
            )
            if self.TO_SAVE_ATTN:
                if self.to_always_save == False:
                    if relevant_models is None:
                        continue
                        # cv2.imwrite(os.path.join(os.getcwd(),'Images',f'{model.name}_{self.image_name[:-4]}_Bare_Attention_image_cv.jpg'), self.last_attn_img_normalized)
                        # cv2.imwrite(os.path.join(os.getcwd(),'Images',f'{model.name}_{self.image_name[:-4]}_Fused_image_cv.jpg'), self.last_fused_img)
                    else:
                        res = any(sub in model.name for sub in relevant_models)
                        if res or output_lp.item() > 0.0:
                            # cv2.imwrite(os.path.join(os.getcwd(),'Images',f'{model.name}_{self.image_name[:-4]}_Bare_Attention_image_cv.jpg'), self.last_attn_img_normalized)
                            cv2.imwrite(
                                os.path.join(
                                    os.getcwd(),
                                    "Images",
                                    f"{self.title}_{model.name}_{self.image_name}_Fused_image_cv.jpg",
                                ),
                                cv2.cvtColor(self.last_fused_img, cv2.COLOR_BGR2RGB),
                            )
                else:
                    cv2.imwrite(
                        os.path.join(
                            os.getcwd(),
                            "Images",
                            f"{self.title}_{model.name}_{self.image_name}_thr_{threshold}_Fused_image_cv.jpg",
                        ),
                        cv2.cvtColor(self.last_fused_img, cv2.COLOR_BGR2RGB),
                    )

        return self.predictions, self.is_from_mobile

    def load_image(self, path, to_cut_image=False):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if to_cut_image:
            img = img[270:1150, :, :]
        self.image_name = path.split("\\")[-1].split(".")[0]
        self.video_inference = False
        width = self.in_w  # Like the cut of NY DB
        height = self.in_h
        dsize = (width, height)
        img = cv2.resize(img, dsize)
        # img = np.transpose(img, (2, 0, 1))
        self.inference_input = img
        self.image = img
        return img


def Execute_Explainability_Analysis():
    # experiment_path_exp1 = r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Writing\Paper 5 - Explainability\Materials\Exp1'
    # experiment_path_exp2 = r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Writing\Paper 5 - Explainability\Materials\Exp2_2'
    # experiment_path_exp2 = r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Writing\Paper 5 - Explainability\Materials\Exp2_corrections'
    # experiment_path_exp3 = r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Writing\Paper 5 - Explainability\Materials\Exp3_2'
    # experiment_path_exp3 = r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Writing\Paper 5 - Explainability\Materials\Exp3_3'
    # experiment_path_exp3 = r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Writing\Paper 5 - Explainability\Materials\Exp3_4'
    # experiment_path_exp3 = r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Writing\Paper 5 - Explainability\Materials\Exp3'
    # experiment_path_RESNET = r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Writing\Paper 5 - Explainability\Materials\Resnet_Mobile'
    # experiment_path_ADDITIONAL_DISEASES = r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Writing\Paper 5 - Explainability\Materials\Additional_diseases'
    experiment_path_PVC_formats = r"C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Writing\Paper 5 - Explainability\Scripts\PVC_images_multiple_formats"
    # Delete all images in folder
    folder = os.path.join(os.getcwd(), "Images")
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                print(f"Deleting file: {file_path}")
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                print(f"Deleting folder: {file_path}")
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))

    # # Experiment 1
    image_path = os.path.join(experiment_path_PVC_formats, "Image_exemplars")
    Draw_Explainability_Image(
        images_path=image_path,
        experiment_path=experiment_path_PVC_formats,
        title="Exp6_",
        figure_name="Exp5",
        to_save=True,
        to_cut_image=False,
        to_use_signal_extraction=False,
    )
    # image_path = os.path.join(experiment_path_exp2, 'Image_exemplars')
    # Draw_Explainability_Image(images_path = image_path, experiment_path = experiment_path_exp1 ,title = 'Exp1_Shadowed_', figure_name = 'Exp1', to_save = True, to_cut_image = False, to_use_signal_extraction = False)
    # Experiment 2
    # image_path = os.path.join(experiment_path_exp2, 'Image_exemplars')
    # Draw_Explainability_Image(images_path = image_path, experiment_path = experiment_path_exp2 ,title = 'Exp2_Shadowed_', figure_name = 'Exp2', to_save = True, to_cut_image = False, to_use_signal_extraction = False, save_signal_image = False)
    # image_path = os.path.join(experiment_path_exp3, 'Image_exemplars')
    # Draw_Explainability_Image(images_path = image_path, experiment_path = experiment_path_exp2 ,title = 'Exp2_Mobile_', figure_name = 'Exp2', to_save = True, to_cut_image = False, to_use_signal_extraction = False)
    # # # Experiment 3
    # image_path = os.path.join(experiment_path_exp1, 'Image_exemplars')
    # Draw_Explainability_Image(images_path = image_path, experiment_path = experiment_path_exp1 ,title = 'Exp1_', figure_name = 'Exp1', to_save = True, to_cut_image = True, to_use_signal_extraction = False)


def Draw_Explainability_Image(
    images_path=None,
    experiment_path=None,
    title=None,
    figure_name=None,
    to_save=True,
    to_cut_image=False,
    to_use_signal_extraction=False,
    relevant_checkpoints_list=None,
    always_save=False,
    save_signal_image=False,
):
    explainability = Explainability_AIO(
        to_use_signal_extraction=to_use_signal_extraction,
        save_signal_extraction=save_signal_image,
    )
    explainability.to_always_save = always_save
    explainability.title = title
    checkpoint_path = os.path.join(experiment_path, "checkpoints")
    explainability.load_inference_models(checkpoint_path)
    # Load CSV file
    legend_path = os.path.join(images_path, "Images_legend.txt")
    legend = pd.read_csv(legend_path, sep="\t")
    # iterate over all images in folder
    for image in os.listdir(images_path):
        if image.endswith(".png"):
            print(f"{title},Processing image: {image}")
            relevant_checkpoints = legend[image]
            relevant_checkpoints = legend["Title"][relevant_checkpoints == 1]
            relevant_checkpoints_list = relevant_checkpoints.values.tolist()
            image_path_ = os.path.join(images_path, image)
            explainability.load_image(image_path_, to_cut_image=to_cut_image)
            explainability.predict(relevant_models=relevant_checkpoints_list)
            # explainability.extract_signal()
        else:
            print(f"{title},Skipping image: {image}")
    print("Finished")


def Test_latest_Resnet():
    # experiment_path_exp1 = r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Writing\Paper 5 - Explainability\Materials\Exp1'
    # experiment_path_exp2 = r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Writing\Paper 5 - Explainability\Materials\Exp2'
    # experiment_path_exp2 = r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Writing\Paper 5 - Explainability\Materials\Exp2_2'
    # experiment_path_exp3 = r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Writing\Paper 5 - Explainability\Materials\Exp3'
    # experiment_path_exp3 = r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Writing\Paper 5 - Explainability\Materials\Exp3_4'
    experiment_path_Resnet = r"C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Writing\Paper 5 - Explainability\Materials\Resnet"
    # experiment_path_Resnet = r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Writing\Paper 5 - Explainability\Materials\Resnet_high_res'
    # experiment_path_Resnet = r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Writing\Paper 5 - Explainability\Materials\Resnet34'
    PVCs_path = r"C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Work\Explainability_Marker\Kenta\PVC"
    image_path = os.path.join(PVCs_path, "Image_exemplars")
    # Delete all images in folder
    folder = os.path.join(os.getcwd(), "Images")
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                print(f"Deleting file: {file_path}")
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                print(f"Deleting folder: {file_path}")
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))
    Draw_Explainability_Image(
        images_path=image_path,
        experiment_path=experiment_path_Resnet,
        title="AIO_4_0_",
        figure_name="AIO_4_0_",
        to_save=True,
        to_cut_image=False,
        to_use_signal_extraction=False,
        relevant_checkpoints_list=None,
        always_save=True,
        save_signal_image=True,
    )

    print("Finished")


def Clinical_images_overlay():
    print("Clinical images overlay")
    experiment_path_AFL_checkpoint = r"C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Writing\Paper 5 - Explainability\Scripts\checkpoints_AFL"
    experiment_path_PVC_checkpoint = r"C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Writing\Paper 5 - Explainability\Scripts\checkpoints_PVC"
    explainability = Explainability_AIO(
        to_use_signal_extraction=False, save_signal_extraction=False
    )
    explainability.to_always_save = True
    explainability.title = "Clinical_"
    checkpoint_path = experiment_path_PVC_checkpoint
    explainability.load_inference_models(checkpoint_path)
    relevant_checkpoints = os.listdir(checkpoint_path)
    image_path_ = r"C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Writing\Paper 5 - Explainability\Scripts\AFL_images"
    gold_image_path = r"C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Work\Explainability_Marker\AFL\Explainability\237.png"
    # image_path_ = r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Writing\Paper 5 - Explainability\Scripts\PVC_images'
    gold_image_path = r"C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Work\Explainability_Marker\PVC\Explainability\237.png"

    i_p_ = image_path_
    for image in os.listdir(image_path_):
        print(f"Processing image: {image}")
        relevant_checkpoints_list = relevant_checkpoints
        image_path_ = os.path.join(i_p_, image)
        explainability.load_image(image_path_, to_cut_image=False)
        explainability.predict(relevant_models=relevant_checkpoints_list)
        beta = 1.0 - explainability.alpha
        original_image = cv2.imread(gold_image_path)
        original_image = cv2.resize(original_image, (1650, 880))
        # show image
        cv2.imshow("image", original_image)
        cv2.waitKey(0)

        Explainability_binary = explainability.last_attn_img_normalized > 20.0
        Kenta_binary = original_image[:, :, 2] > 245.0
        Explainability_binary = Explainability_binary[:, :, 0]
        Kenta_binary[1:, :] = Kenta_binary[1:, :] + Kenta_binary[:-1, :]
        Kenta_binary[:, 1:] = Kenta_binary[:, 1:] + Kenta_binary[:, :-1]
        Kenta_binary = Kenta_binary > 0
        Explainability_binary[1:, :] = (
            Explainability_binary[1:, :] + Explainability_binary[:-1, :]
        )
        Explainability_binary[:, 1:] = (
            Explainability_binary[:, 1:] + Explainability_binary[:, :-1]
        )
        Explainability_binary = Explainability_binary > 0
        Positive_overlap = Explainability_binary == Kenta_binary
        Positive_overlap[Kenta_binary == False] = False
        Positive_overlap[Explainability_binary == False] = False

        # Positive_overlap[1:,:]+= (Explainability_binary[1:,:] ==Kenta_binary[:-1,:] ) + (Explainability_binary[:,1:] ==Kenta_binary[:,:-1] )
        False_positive = (Explainability_binary == True) * (Kenta_binary == False)
        Merged_image = original_image.copy()
        Merged_image[Positive_overlap] = (255, 0, 0)
        Merged_image[False_positive] = (0, 255, 0)
        # Emphasize overlap
        # show image
        cv2.imshow("image", Merged_image)
        cv2.waitKey(0)
        expl_img = explainability.last_attn_img_normalized.copy()
        expl_img[expl_img[:, :, 0] < 20] = (0, 0, 0)

        last_fused_img = cv2.addWeighted(
            original_image, explainability.alpha, expl_img, beta, 0.0
        )
        last_fused_img = increase_brightness(last_fused_img, value=100)
        cv2.imshow("image fused", last_fused_img)
        cv2.waitKey(0)
        cv2.imwrite(
            os.path.join(
                os.getcwd(),
                "Images",
                f"{explainability.title}_{image[:-4]}_Fused_with_Kenta_cv.jpg",
            ),
            cv2.cvtColor(last_fused_img, cv2.COLOR_BGR2RGB),
        )


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def Correlation_analysis():
    print("Correlation_analysis")
    experiment_path_AFL_checkpoint = r"C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Writing\Paper 5 - Explainability\Scripts\checkpoints_AFL"
    # experiment_path_PVC_checkpoint = r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Writing\Paper 5 - Explainability\Scripts\checkpoints_PVC'
    explainability = Explainability_AIO(
        to_use_signal_extraction=False, save_signal_extraction=False
    )
    explainability.to_always_save = True
    explainability.title = "Clinical_"
    # checkpoint_path = experiment_path_PVC_checkpoint
    checkpoint_path = experiment_path_AFL_checkpoint
    explainability.load_inference_models(checkpoint_path)
    relevant_checkpoints = os.listdir(checkpoint_path)

    # image_path_ = r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Work\Explainability_Marker\Kenta\PVC'
    # gold_image_path = r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Work\Explainability_Marker\PVC\Explainability'

    image_path_ = r"C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Work\Explainability_Marker\Kenta\AFL"
    gold_image_path = r"C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Work\Explainability_Marker\AFL\Explainability"

    i_p_ = image_path_
    for image in os.listdir(image_path_):
        print(f"Processing image: {image}")
        relevant_checkpoints_list = relevant_checkpoints
        image_path_ = os.path.join(i_p_, image)
        gold_image_path_ = os.path.join(gold_image_path, image)
        if os.path.exists(gold_image_path_) == False:
            continue
        explainability.load_image(image_path_, to_cut_image=False)
        explainability.predict(relevant_models=relevant_checkpoints_list)
        beta = 1.0 - explainability.alpha
        original_image = cv2.imread(image_path_)
        original_image = cv2.resize(original_image, (1650, 880))
        gold_image = cv2.imread(os.path.join(gold_image_path, image))
        gold_image = cv2.resize(gold_image, (1650, 880))
        # show image
        # cv2.imshow('image',original_image)
        # cv2.waitKey(0)
        # cv2.imshow('image',gold_image)
        # cv2.waitKey(0)

        Explainability_binary = explainability.last_attn_img_normalized > 20.0
        Kenta_binary = (
            (gold_image[:, :, 2] > 245.0)
            * (gold_image[:, :, 1] < 245.0)
            * (gold_image[:, :, 0] < 245.0)
        )
        Explainability_binary = Explainability_binary[:, :, 0]
        Kenta_binary[1:, :] = Kenta_binary[1:, :] + Kenta_binary[:-1, :]
        Kenta_binary[:, 1:] = Kenta_binary[:, 1:] + Kenta_binary[:, :-1]
        Kenta_binary = Kenta_binary > 0
        Explainability_binary[1:, :] = (
            Explainability_binary[1:, :] + Explainability_binary[:-1, :]
        )
        Explainability_binary[:, 1:] = (
            Explainability_binary[:, 1:] + Explainability_binary[:, :-1]
        )
        Explainability_binary = Explainability_binary > 0
        Positive_overlap = Explainability_binary == Kenta_binary
        shape_ = Positive_overlap.shape
        with open("Coefficients_AFL.txt", "a") as f:
            f.write(f"{image},{np.sum(Positive_overlap)/(shape_[0]*shape_[1])}\n")
        Positive_overlap[Kenta_binary == False] = False
        Positive_overlap[Explainability_binary == False] = False

        # Positive_overlap[1:,:]+= (Explainability_binary[1:,:] ==Kenta_binary[:-1,:] ) + (Explainability_binary[:,1:] ==Kenta_binary[:,:-1] )
        False_positive = (Explainability_binary == True) * (Kenta_binary == False)
        Merged_image = original_image.copy()
        Merged_image[Positive_overlap] = (255, 0, 0)
        Merged_image[False_positive] = (0, 255, 0)
        # Emphasize overlap
        # show image
        cv2.imshow("image", Merged_image)
        cv2.waitKey(0)
        expl_img = explainability.last_attn_img_normalized.copy()
        expl_img[expl_img[:, :, 0] < 20] = (0, 0, 0)

        last_fused_img = cv2.addWeighted(
            original_image, explainability.alpha, expl_img, beta, 0.0
        )
        last_fused_img = increase_brightness(last_fused_img, value=100)
        cv2.imshow("image fused", last_fused_img)
        cv2.waitKey(0)
        cv2.imwrite(
            os.path.join(
                os.getcwd(),
                "Images",
                f"{explainability.title}_{image[:-4]}_Fused_with_Kenta_cv.jpg",
            ),
            cv2.cvtColor(last_fused_img, cv2.COLOR_BGR2RGB),
        )


def Disease_absence_example():
    print("Disease_absence_example")
    experiment_path_AFIB_checkpoint = r"C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Writing\Paper 5 - Explainability\Scripts\checkpoints_AFIB"
    explainability = Explainability_AIO(
        to_use_signal_extraction=False, save_signal_extraction=False
    )
    explainability.to_always_save = True
    explainability.title = "Absence_of_cardiac_disorder"
    # checkpoint_path = experiment_path_PVC_checkpoint
    checkpoint_path = experiment_path_AFIB_checkpoint
    explainability.load_inference_models(checkpoint_path)
    relevant_checkpoints = os.listdir(checkpoint_path)

    # image_path_ = r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Work\Explainability_Marker\Kenta\PVC'
    # gold_image_path = r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Work\Explainability_Marker\PVC\Explainability'

    image_path_ = r"C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Writing\Paper 5 - Explainability\Scripts\NON_AFIB_images"

    i_p_ = image_path_
    for image in os.listdir(i_p_):
        for thr in range(0, 30, 5):
            explainability.threshold = thr
            print(f"Processing image: {image}")
            relevant_checkpoints_list = relevant_checkpoints
            image_path_ = os.path.join(i_p_, image)
            explainability.load_image(image_path_, to_cut_image=False)
            explainability.predict(relevant_models=relevant_checkpoints_list)
            beta = 1.0 - explainability.alpha
            original_image = cv2.imread(image_path_)
            original_image = cv2.resize(original_image, (1650, 880))


if __name__ == "__main__":
    # Disease_absence_example()
    # Clinical_images_overlay()
    # Correlation_analysis()
    # print("Assessment of explainability methods")
    # Test_latest_Resnet()
    Execute_Explainability_Analysis()
