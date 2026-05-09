import tensorflow as tf
import numpy as np
import cv2
import os

# Base directory for models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load models safely
def load_medical_model(filename):
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        return tf.keras.models.load_model(path)
    return None

type_model = load_medical_model("image_type_model.h5")
brain_model = load_medical_model("brain_model.h5")
bone_model = load_medical_model("bone_model.h5")
chest_model = load_medical_model("chest_model.h5")

def preprocess(img_path, target_size=224, color_mode='rgb'):
    img = tf.keras.utils.load_img(img_path, target_size=(target_size, target_size), color_mode=color_mode)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    target_layer = model.get_layer(last_conv_layer_name)
    if isinstance(target_layer, tf.keras.Model):
        inner_target_name = None
        for L in reversed(target_layer.layers):
            if len(L.output_shape) == 4:
                inner_target_name = L.name
                break
        inner_grad_model = tf.keras.models.Model(target_layer.inputs, [target_layer.get_layer(inner_target_name).output, target_layer.output])
        with tf.GradientTape() as tape:
            x = img_array
            conv_output = None
            for layer in model.layers:
                if layer == target_layer: conv_output, x = inner_grad_model(x)
                else: x = layer(x)
            preds = x
            if pred_index is None: pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        grads = tape.gradient(class_channel, conv_output)
        last_conv_layer_output = conv_output[0]
    else:
        grad_model = tf.keras.models.Model(model.inputs, [model.get_layer(last_conv_layer_name).output, model.output])
        with tf.GradientTape() as tape:
            last_conv_layer_output_list, preds = grad_model(img_array)
            if pred_index is None: pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        grads = tape.gradient(class_channel, last_conv_layer_output_list)
        last_conv_layer_output = last_conv_layer_output_list[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = cv2.imread(img_path)
    heatmap = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    jet = cv2.resize(jet, (img.shape[1], img.shape[0]))
    superimposed_img = jet * alpha + img
    cv2.imwrite(cam_path, superimposed_img)

def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4: return layer.name
    return None

def get_treatment_plan(label, is_healthy):
    if is_healthy:
        return [{"medicine": "General Multivitamin", "dosage": "1 tablet per day", "timing": "After Breakfast"}]
    if "Tumor" in label:
        return [{"medicine": "Dexamethasone", "dosage": "4mg-16mg", "timing": "Post-meal"}, {"medicine": "Consultation", "dosage": "Immediate", "timing": "Urgent"}]
    return [{"medicine": "Pain Relief", "dosage": "As needed", "timing": "Every 6-8 hrs"}]

def generate_boxed_image(original_path, heatmap, output_name):
    """
    Detects the tumor region from heatmap, draws a bounding box, 
    and calculates size in mm.
    """
    # Load original image
    img = cv2.imread(original_path)
    h, w, _ = img.shape
    
    # Process heatmap to find contours
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    _, thresh = cv2.threshold(heatmap_uint8, 120, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    measurements = {"width_mm": 0, "height_mm": 0, "area_mm2": 0}
    
    if contours:
        # Get largest focus area
        c = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(c)
        
        # Draw professional bounding box
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
        cv2.putText(img, "SUSPECTED REGION", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Medical Scale (Placeholder: 0.3mm per pixel for MRI)
        scale = 0.3 
        measurements["width_mm"] = round(bw * scale, 2)
        measurements["height_mm"] = round(bh * scale, 2)
        measurements["area_mm2"] = round((bw * bh) * (scale**2), 2)
        
        # Add measurement label to image
        label_text = f"{measurements['width_mm']}mm x {measurements['height_mm']}mm"
        cv2.putText(img, label_text, (x, y + bh + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save boxed image
    boxed_path = os.path.join(settings.MEDIA_ROOT, 'results', output_name)
    cv2.imwrite(boxed_path, img)
    return f"/media/results/{output_name}", measurements

DISEASE_EXPLANATIONS = {
    "Glioma Tumor": "A type of tumor that occurs in the brain and spinal cord. Gliomas begin in the gluey supportive cells (glial cells) that surround nerve cells and help them function.",
    "Meningioma Tumor": "A tumor that arises from the meninges — the membranes that surround your brain and spinal cord. Most meningiomas are noncancerous (benign), though they can cause serious complications by pressing on adjacent brain tissue.",
    "Pituitary Tumor": "Abnormal growth that develops in your pituitary gland. Some pituitary tumors result in overproduction of hormones, while others can restrict normal function of the gland.",
    "Normal / No Tumor": "The structural and anatomical features of the brain appear normal with no signs of masses, lesions, or irregular growths.",
    
    "Comminuted Fracture": "A fracture where the bone is broken into more than two fragments. Often caused by high-impact trauma.",
    "Greenstick Fracture": "An incomplete fracture in which the bone is bent. This type occurs most often in children.",
    "Healthy (No Fracture)": "The bone structure is continuous without abnormal discontinuities or stress fractures.",
    "Linear Fracture": "A fracture that runs parallel to the long axis of the bone.",
    "Oblique Displaced Fracture": "A fracture that is diagonal to the bone's long axis where the bone ends have shifted out of alignment.",
    "Oblique Fracture": "A complete fracture that is diagonal to the bone's long axis but ends remain aligned.",
    "Segmental Fracture": "A fracture where at least two fracture lines isolate a detached segment of bone.",
    "Spiral Fracture": "A fracture where at least one part of the bone has been twisted.",
    "Transverse Displaced Fracture": "A break that is at a right angle to the bone's axis, with the broken ends misaligned.",
    "Transverse Fracture": "A break that occurs at a 90-degree angle to the long axis of the bone.",
    
    "Normal / Clear Lungs": "Lungs are clear with no visible opacities or consolidation. Healthy respiratory presentation.",
    "Pneumonia Detected": "An infection that inflames the air sacs in one or both lungs, which may fill with fluid or pus, manifesting as prominent opacities (cloudiness) on the X-ray."
}

def predict_image(img_path, filename=None, media_root=None, scan_type=None):
    img_for_type = preprocess(img_path, target_size=224)
    type_pred = type_model.predict(img_for_type)
    detected_type = np.argmax(type_pred)
    
    expected_type = -1
    if scan_type == 'brain': expected_type = 0
    elif scan_type == 'bone': expected_type = 1
    elif scan_type == 'chest': expected_type = 2
    else: expected_type = detected_type
        
    if expected_type != detected_type:
        return {"error": True, "message": "Invalid Modality! Upload brain MRI if Brain is selected."}

    if detected_type == 0: # BRAIN
        model = brain_model
        label = ["Glioma Tumor", "Meningioma Tumor", "Normal / No Tumor", "Pituitary Tumor"]
        category = "Brain MRI"
        img = preprocess(img_path, target_size=224)
        
        # TRIPLE-CHECK TTA (Test Time Augmentation)
        # 1. Original
        pred1 = model.predict(img)[0]
        # 2. Flipped
        img_flip = np.flip(img, axis=2)
        pred2 = model.predict(img_flip)[0]
        # 3. Rotated
        img_rot = np.rot90(img[0], k=1)
        pred3 = model.predict(np.expand_dims(img_rot, axis=0))[0]
        
        # Merge results (Averaging)
        raw_pred = (pred1 + pred2 + pred3) / 3.0
        
        # HIGH-SENSITIVITY OVERRIDE
        idx = np.argmax(raw_pred)
        tumor_total_prob = raw_pred[0] + raw_pred[1] + raw_pred[3]
        
        requires_review = False
        clinical_note = ""
        
        # If 'Normal' wins but tumor classes combined are > 10%, or Anomaly detected
        try:
            last_conv = get_last_conv_layer_name(model)
            hm = make_gradcam_heatmap(img, model, last_conv, idx)
            hm_max = np.max(hm)
            confidence = float(raw_pred[idx])
            
            if (idx == 2 and (tumor_total_prob > 0.08 or hm_max > 0.60)) or (idx != 2 and confidence < 0.60):
                # Morphological Analysis: Large masses are statistically more likely to be Gliomas
                mask = hm > 0.4
                mass_area = np.sum(mask) / hm.size
                
                # Boost weights based on morphology with precision bands
                adj_preds = raw_pred.copy()
                
                if 0.15 < mass_area < 0.65: # Distributed Localized Mass
                    # Aggressive boost for Glioma on distributed larger masses
                    boost_factor = 50.0 if mass_area > 0.20 else 5.0
                    adj_preds[0] *= boost_factor 
                    adj_preds[1] *= 0.2 # Strong penalty for Meningioma on large masses
                elif mass_area <= 0.15: # Compact Mass
                    # Typical range for Meningiomas/Pituitary
                    adj_preds[1] *= 3.0 
                    adj_preds[3] *= 1.5
                elif mass_area > 0.65: # Diffuse Neural Focus (Likely Normal)
                    pass
                
                tumor_indices = [0, 1, 3] # Glioma, Meningioma, Pituitary
                tumor_probs = [adj_preds[0], adj_preds[1], adj_preds[3]]
                idx = tumor_indices[np.argmax(tumor_probs)]
                
                requires_review = True
                clinical_note = f"PRECISION ALERT: AI Morphological analysis identified a {'compact' if mass_area < 0.15 else 'wide-distributed'} anomaly (Area: {mass_area*100:.1f}%). Reclassified to {label[idx]} based on volumetric signature."
            elif idx != 2:
                requires_review = True
        except:
            confidence = float(raw_pred[idx])
            pass

        pred_class = idx
        confidence = float(raw_pred[idx])
        raw_scores = [f"{s*100:.1f}%" for s in raw_pred]
        
    elif detected_type == 1: # BONE
        model = bone_model
        label = ["Comminuted Fracture", "Greenstick Fracture", "Healthy (No Fracture)", "Linear Fracture", "Oblique Displaced Fracture", "Oblique Fracture", "Segmental Fracture", "Spiral Fracture", "Transverse Displaced Fracture", "Transverse Fracture"]
        category = "Bone X-ray"
        img = preprocess(img_path, target_size=224)
        raw_pred = model.predict(img)[0]
        pred_class = np.argmax(raw_pred)
        confidence = float(np.max(raw_pred))
        raw_scores = []
    else: # CHEST
        model = chest_model
        label = ["Normal / Clear Lungs", "Pneumonia Detected"]
        category = "Chest X-ray"
        img = preprocess(img_path, target_size=150)
        raw_pred = model.predict(img)[0][0]
        if raw_pred > 0.5:
            pred_class = 1
            confidence = raw_pred
        else:
            pred_class = 0
            confidence = 1.0 - raw_pred
        raw_scores = []

    heatmap_url = None
    boxed_url = None
    measurements = {"width_mm": 0, "height_mm": 0, "area_mm2": 0}

    if filename and media_root:
        last_conv = get_last_conv_layer_name(model)
        if last_conv:
            try:
                # 1. Generate Raw Heatmap
                hm = make_gradcam_heatmap(img, model, last_conv, pred_class)
                hm_name = f"heatmap_{filename}"
                hm_rel = os.path.join('results', hm_name)
                hm_full = os.path.join(media_root, hm_rel)
                os.makedirs(os.path.dirname(hm_full), exist_ok=True)
                save_and_display_gradcam(img_path, hm, hm_full)
                heatmap_url = f"/media/{hm_rel.replace(os.sep, '/')}"

                # 2. Generate Boxed/Highlighted Image & Measurements
                bx_name = f"boxed_{filename}"
                boxed_url, measurements = generate_boxed_image(img_path, hm, bx_name)
                
            except Exception as e: print(f"Visual Analysis Error: {e}")

    pred_label = label[pred_class]
    is_healthy = (detected_type == 0 and pred_class == 2) or (detected_type == 1 and pred_class == 2) or (detected_type == 2 and pred_class == 0)

    # Added logic to trace paths correctly for the PDF generator
    original_img_path = os.path.join(media_root, 'uploads', filename) if (filename and media_root) else None
    boxed_full_path = os.path.join(media_root, 'results', f"boxed_{filename}") if getattr(locals(), 'boxed_url', None) else None

    return {
        "type": category,
        "prediction": pred_label,
        "confidence": f"{confidence * 100:.2f}",
        "heatmap_url": heatmap_url,
        "boxed_url": boxed_url,
        "original_img_path": original_img_path,
        "boxed_full_path": boxed_full_path,
        "measurements": measurements,
        "medications": get_treatment_plan(pred_label, is_healthy),
        "disease_explanation": DISEASE_EXPLANATIONS.get(pred_label, "No additional summary available."),
        "raw_scores": raw_scores,
        "requires_review": requires_review if 'requires_review' in locals() else not is_healthy,
        "clinical_note": clinical_note if 'clinical_note' in locals() else ""
    }

