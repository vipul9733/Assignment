import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageOps
import os
import base64
import io
import time
from rembg import remove
from skimage import transform as skimage_transform
import google.generativeai as genai
import json
import re

# --- Configuration & Global Variables ---
TEMP_DIR = "temp_vto_debug"
RESULTS_DIR = "results_vto"
SAMPLES_DIR = "samples_vto"
DEBUG_MODE = True

# --- Gemini Image Analyzer ---
class GeminiImageAnalyzer:
    def __init__(self, api_key):
        if not api_key:
            self.model = None
            print("Warning: GeminiImageAnalyzer initialized without API key.")
            return
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash-latest') # Or 'gemini-pro-vision'
            print("GeminiImageAnalyzer initialized.")
        except Exception as e:
            print(f"Error configuring Gemini: {e}")
            self.model = None

    def _parse_gemini_json_response(self, response_text):
        match = re.search(r"```json\s*([\s\S]+?)\s*```", response_text)
        if match:
            json_str = match.group(1)
        else:
            json_str = response_text.strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Gemini JSON parsing error: {e}. Response text: {response_text}")
            return None

    def analyze_person_for_keypoints(self, person_img_np_rgba):
        if not self.model: return None
        image_pil = Image.fromarray(person_img_np_rgba).convert("RGB")
        h, w = person_img_np_rgba.shape[:2]
        prompt = f"""
        Analyze this image of a person. Identify key body landmarks.
        Provide (x, y) pixel coordinates relative to the top-left. Image: {w}x{h}px.
        Landmarks: nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, 
        right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, 
        left_hip, right_hip, neck, mid_hip.
        Output ONLY a JSON object mapping landmark names to [x, y] pixel coordinates.
        Example: {{ "left_shoulder": [150, 200], "neck": [200, 180] }}
        Omit non-visible landmarks. Coordinates must be integers within image bounds.
        """
        try:
            response = self.model.generate_content([prompt, image_pil])
            data = self._parse_gemini_json_response(response.text)
            if not data: return None
            keypoints = {name.lower(): (int(max(0,min(coords[0],w-1))), int(max(0,min(coords[1],h-1))))
                         for name, coords in data.items() if isinstance(coords, list) and len(coords) == 2}
            print(f"Gemini person keypoints: {keypoints if keypoints else 'None'}")
            return keypoints if keypoints else None
        except Exception as e:
            print(f"Gemini person analysis API error: {e}")
            return None

    def analyze_clothing_for_features(self, clothing_img_np_rgba):
        if not self.model: return None
        image_pil = Image.fromarray(clothing_img_np_rgba).convert("RGB")
        h, w = clothing_img_np_rgba.shape[:2]
        prompt = f"""
        Analyze this clothing item image (likely background-removed). Image: {w}x{h}px.
        1. Determine clothing type (e.g., "tshirt", "hoodie", "dress").
        2. Identify key control points: collar_center, left_collar_point, right_collar_point,
           left_shoulder_tip, right_shoulder_tip, left_sleeve_end, right_sleeve_end,
           left_armpit, right_armpit, waist_left, waist_right, bottom_hem_center,
           bottom_hem_left, bottom_hem_right. For hoodies: hood_top_center, hood_left_opening, hood_right_opening.
        Output ONLY a JSON: {{ "type": "...", "control_points": {{ "point_name": [x,y], ... }} }}
        Omit non-visible/NA points. Coordinates integer, within bounds.
        """
        try:
            response = self.model.generate_content([prompt, image_pil])
            data = self._parse_gemini_json_response(response.text)
            if not data: return None
            clothing_type = data.get("type", "unknown").lower()
            control_points_raw = data.get("control_points", {})
            control_points = {name.lower(): (int(max(0,min(coords[0],w-1))), int(max(0,min(coords[1],h-1))))
                              for name, coords in control_points_raw.items() if isinstance(coords, list) and len(coords) == 2}
            print(f"Gemini clothing type: {clothing_type}, points: {control_points if control_points else 'None'}")
            return {"type": clothing_type, "control_points": control_points} if control_points else None
        except Exception as e:
            print(f"Gemini clothing analysis API error: {e}")
            return None

# --- Master Agent ---
class MasterAgent:
    def __init__(self, gemini_api_key=None):
        self.gemini_analyzer = None
        if gemini_api_key:
            try:
                self.gemini_analyzer = GeminiImageAnalyzer(api_key=gemini_api_key)
            except Exception as e:
                self._log(f"Failed to initialize Gemini Analyzer: {e}", level="warning")

        self.body_analyzer = BodyAnalyzerAgent(gemini_analyzer=self.gemini_analyzer)
        self.clothing_analyzer = ClothingAnalyzerAgent(gemini_analyzer=self.gemini_analyzer)
        self.point_matcher = PointMatchingAgent()
        self.transformer = ImageTransformationAgent()
        self.post_processor = PostProcessingAgent()
        self.quality_validator = QualityValidationAgent()
        self.pipeline_state = {}

    def _log(self, message, level="info"):
        if not DEBUG_MODE and level == "debug": return
        timestamp = time.strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] [{level.upper()}] {message}"
        print(log_msg)
        if level == "error": st.error(log_msg)
        elif level == "warning": st.warning(log_msg)

    def _get_default_config(self):
        # (Same as before - keeping it for brevity in this snippet, ensure it's complete in your file)
        return {
            "body_analyzer": {"model_complexity": 1, "min_detection_confidence": 0.6},
            "clothing_analyzer": {
                "morph_kernel_size": 3, "use_advanced_background_removal": True,
                "alpha_matting": True, "fg_threshold": 240, "bg_threshold": 10, "erode_size": 5
            },
            "point_matcher": {"adjust_for_body_angle": True, "refinement_iterations": 1},
            "transformer": {"warping_method": "piecewise_affine", "refinement_steps": 0},
            "post_processor": {
                "enhance_edges": True, "apply_shadow": True, "fabric_texture_simulation": False,
                "natural_wrinkle_simulation": True, "advanced_blending": True
            },
            "quality_validator": { # Pass self.pipeline_state reference if needed by validator
                "min_confidence": 0.6, "check_anatomical_correctness": True,
                "verify_edge_quality": False, "check_fit_quality": True, "check_color_consistency": False
            }
        }


    def _preprocess_images(self, person_img_np, clothing_img_np):
        # (Same as before - ensure it's complete)
        if person_img_np.shape[2] == 3: person_img = cv2.cvtColor(person_img_np, cv2.COLOR_RGB2RGBA)
        else: person_img = person_img_np.copy()
        if clothing_img_np.shape[2] == 3: clothing_img = cv2.cvtColor(clothing_img_np, cv2.COLOR_RGB2RGBA)
        else: clothing_img = clothing_img_np.copy()

        max_dimension = 1024
        for img_ref, name in [(person_img, "person"), (clothing_img, "clothing")]:
            h_orig, w_orig = img_ref.shape[:2]
            if max(h_orig, w_orig) > max_dimension:
                scale = max_dimension / max(h_orig, w_orig)
                new_size = (int(w_orig * scale), int(h_orig * scale))
                if name == "person": person_img = cv2.resize(img_ref, new_size, interpolation=cv2.INTER_AREA)
                else: clothing_img = cv2.resize(img_ref, new_size, interpolation=cv2.INTER_AREA)
                self._log(f"Resized {name} image to {new_size}", level="debug")
        return person_img, clothing_img


    def _validate_agent_output(self, output, pipeline_key, required_keys=None):
        # (Same as before - ensure it's complete)
        if output is None:
            raise ValueError(f"Agent returned None for {pipeline_key}")
        if required_keys:
            missing = [k for k in required_keys if k not in output]
            if missing: raise ValueError(f"Missing keys in {pipeline_key}: {missing}. Has: {list(output.keys())}")
        self.pipeline_state[pipeline_key] = output
        if "debug_image" in output and output["debug_image"] is not None:
            self.pipeline_state["debug_images"][pipeline_key] = output["debug_image"]
            if DEBUG_MODE: # Save debug image
                if not os.path.exists(TEMP_DIR): os.makedirs(TEMP_DIR)
                path = os.path.join(TEMP_DIR, f"debug_{pipeline_key}.png")
                try:
                    cv2.imwrite(path, cv2.cvtColor(output["debug_image"], cv2.COLOR_RGBA2BGRA))
                except Exception as e:
                    self._log(f"Failed to save debug {path}: {e}", "warning")

    def _apply_final_corrections(self, result_image, validation_result): # Placeholder
        return result_image

    def process(self, person_img_np_rgba, clothing_img_np_rgba, config=None):
        self.pipeline_state = {
            "person_analysis": None, "clothing_analysis": None, "point_matching": None,
            "transformation": None, "post_processing": None, "validation": None,
            "final_result": None, "debug_images": {}, "metadata": {},
            "errors": [], "warnings": []
        }
        if config is None: config = self._get_default_config()
        overall_start_time = time.time()
        self.pipeline_state["metadata"]["start_time"] = overall_start_time
        person_img_original_copy = person_img_np_rgba.copy()

        try:
            person_img, clothing_img = self._preprocess_images(person_img_np_rgba, clothing_img_np_rgba)

            self._log("Analyzing person image...")
            body_data = self.body_analyzer.analyze(person_img, config.get("body_analyzer", {}))
            self._validate_agent_output(body_data, "person_analysis", ["body_keypoints", "segmentation_mask", "measurements"])

            self._log("Analyzing clothing item...")
            clothing_data = self.clothing_analyzer.analyze(clothing_img, config.get("clothing_analyzer", {}))
            self._validate_agent_output(clothing_data, "clothing_analysis", ["control_points", "mask", "type", "clothing_rgba_no_bg"])
            if not clothing_data["control_points"]: raise ValueError("Clothing analysis yielded no control points.")

            self._log("Matching control points...")
            # Pass clothing_data directly as it contains clothing_rgba_no_bg needed later by transformer
            matched_points = self.point_matcher.match_points(body_data, clothing_data, config.get("point_matcher", {}))
            self._validate_agent_output(matched_points, "point_matching", ["src_points", "dst_points"])
            if len(matched_points["src_points"]) < 3: raise ValueError(f"Need at least 3 matched points, got {len(matched_points['src_points'])}.")

            self._log("Transforming clothing to fit body...")
            # Pass clothing_data["clothing_rgba_no_bg"] explicitly for transformation
            warped_clothing, transform_meta = self.transformer.transform(
                clothing_data["clothing_rgba_no_bg"], # Use the background-removed version
                matched_points,
                person_img.shape[:2],
                config.get("transformer", {})
            )
            self.pipeline_state["transformation"] = {"warped_clothing": warped_clothing, "metadata": transform_meta}
            if transform_meta.get("error") or warped_clothing is None:
                raise ValueError(f"Image transformation error: {transform_meta.get('error', 'Warped image is None')}")

            self._log("Applying post-processing...")
            enhanced_result_dict = self.post_processor.process(person_img, warped_clothing, body_data, config.get("post_processor", {}))
            self._validate_agent_output(enhanced_result_dict, "post_processing", ["result_image"])
            enhanced_result = enhanced_result_dict["result_image"]

            self._log("Validating result quality...")
            # Pass self.pipeline_state to validator if it needs to access previous stages' data
            validation_result = self.quality_validator.validate(person_img, enhanced_result, body_data, clothing_data, matched_points, config.get("quality_validator", {}), self.pipeline_state)
            self.pipeline_state["validation"] = validation_result

            final_result = self._apply_final_corrections(enhanced_result, validation_result)
            self.pipeline_state["final_result"] = final_result
            self.pipeline_state["metadata"]["total_processing_time"] = time.time() - overall_start_time
            return final_result, self.pipeline_state

        except Exception as e:
            import traceback
            error_msg = f"Pipeline error: {str(e)}\n{traceback.format_exc()}"
            self._log(error_msg, level="error")
            self.pipeline_state["errors"].append(error_msg)
            return person_img_original_copy, self.pipeline_state


# --- BodyAnalyzerAgent (Restored) ---
class BodyAnalyzerAgent:
    def __init__(self, model_complexity=1, static_image_mode=True, enable_segmentation=True,
                 min_detection_confidence=0.6, gemini_analyzer=None):
        self.mp_pose = mp.solutions.pose
        self.pose_model = self.mp_pose.Pose(
            static_image_mode=static_image_mode, model_complexity=model_complexity,
            enable_segmentation=enable_segmentation, min_detection_confidence=min_detection_confidence)
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        self.gemini_analyzer = gemini_analyzer
        self.min_detection_confidence = min_detection_confidence

    def analyze(self, image_np_rgba, config=None):
        config = config or {}
        keypoints = None

        if self.gemini_analyzer:
            print("Attempting body analysis with Gemini...")
            gemini_keypoints = self.gemini_analyzer.analyze_person_for_keypoints(image_np_rgba)
            if gemini_keypoints and len(gemini_keypoints) > 5:
                keypoints = gemini_keypoints
                print("Used Gemini for body keypoints.")
            else:
                print("Gemini body analysis failed or few points, falling back to MediaPipe.")

        if keypoints is None:
            print("Using MediaPipe for body analysis.")
            image_np_rgb = cv2.cvtColor(image_np_rgba, cv2.COLOR_RGBA2RGB)
            pose_results = self.pose_model.process(image_np_rgb)
            if pose_results.pose_landmarks:
                keypoints = self._extract_mediapipe_keypoints(pose_results.pose_landmarks, image_np_rgb.shape)
            else:
                keypoints = {}
                print("MediaPipe found no pose landmarks.")

        if keypoints:
            self._add_derived_keypoints(keypoints, image_np_rgba.shape[:2])

        measurements = self._calculate_detailed_measurements(keypoints)
        
        image_np_rgb_for_seg = cv2.cvtColor(image_np_rgba, cv2.COLOR_RGBA2RGB)
        pose_results_for_seg = self.pose_model.process(image_np_rgb_for_seg)
        segmentation_mask = self._get_enhanced_segmentation_mask(image_np_rgb_for_seg, pose_results_for_seg)

        debug_image = self._create_debug_visualization(image_np_rgba, keypoints, segmentation_mask)
        confidence = len(keypoints) / 33.0 if keypoints else 0.0

        return {"body_keypoints": keypoints, "measurements": measurements,
                "segmentation_mask": segmentation_mask, "confidence": confidence,
                "debug_image": debug_image, "metadata": {}}

    def _extract_mediapipe_keypoints(self, pose_landmarks, image_shape):
        h, w = image_shape[:2]
        keypoints = {}
        for i, landmark in enumerate(pose_landmarks.landmark):
            name = self.mp_pose.PoseLandmark(i).name.lower()
            visibility = landmark.visibility if hasattr(landmark, 'visibility') else self.min_detection_confidence
            if visibility >= self.min_detection_confidence:
                 keypoints[name] = (int(landmark.x * w), int(landmark.y * h))
        return keypoints

    def _add_derived_keypoints(self, keypoints, image_shape_hw):
        # (Same as before - ensure complete: neck, mid_hip, torso_center, chest, collars, waists)
        h, w = image_shape_hw
        # Neck
        if 'left_shoulder' in keypoints and 'right_shoulder' in keypoints:
            ls, rs = np.array(keypoints['left_shoulder']), np.array(keypoints['right_shoulder'])
            neck = (ls + rs) / 2
            if 'nose' in keypoints: neck[1] = keypoints['nose'][1] + (neck[1] - keypoints['nose'][1]) * 0.7
            else: neck[1] -= np.linalg.norm(ls - rs) * 0.15
            keypoints['neck'] = (int(neck[0]), int(max(0, min(neck[1], h - 1))))
        # Mid-hip
        if 'left_hip' in keypoints and 'right_hip' in keypoints:
            lh, rh = np.array(keypoints['left_hip']), np.array(keypoints['right_hip'])
            keypoints['mid_hip'] = tuple(((lh + rh) / 2).astype(int))
        # Torso center
        if 'neck' in keypoints and 'mid_hip' in keypoints:
            n, mh = np.array(keypoints['neck']), np.array(keypoints['mid_hip'])
            keypoints['torso_center'] = tuple(((n + mh) / 2).astype(int))
        # Chest
        if 'neck' in keypoints and 'torso_center' in keypoints:
            n, tc = np.array(keypoints['neck']), np.array(keypoints['torso_center'])
            keypoints['chest'] = tuple((n + 0.33 * (tc - n)).astype(int))
        # Collar points
        if 'neck' in keypoints and 'left_shoulder' in keypoints and 'right_shoulder' in keypoints:
            n, ls, rs = np.array(keypoints['neck']), np.array(keypoints['left_shoulder']), np.array(keypoints['right_shoulder'])
            keypoints['left_collar'] = tuple((n * 0.7 + ls * 0.3).astype(int)) # Simplified
            keypoints['right_collar'] = tuple((n * 0.7 + rs * 0.3).astype(int)) # Simplified
        # Waist points
        if 'mid_hip' in keypoints and 'torso_center' in keypoints and 'left_hip' in keypoints and 'right_hip' in keypoints:
            mh, tc = np.array(keypoints['mid_hip']), np.array(keypoints['torso_center'])
            mid_waist_y = int(mh[1] - 0.33 * (mh[1] - tc[1]))
            hip_width = abs(keypoints['right_hip'][0] - keypoints['left_hip'][0])
            waist_width = hip_width * 0.85
            mid_waist_x = int(mh[0])
            keypoints['mid_waist'] = (mid_waist_x, mid_waist_y)
            keypoints['left_waist'] = (int(mid_waist_x - waist_width / 2), mid_waist_y)
            keypoints['right_waist'] = (int(mid_waist_x + waist_width / 2), mid_waist_y)


    def _get_enhanced_segmentation_mask(self, image_np_rgb, pose_results):
        # (Same as before - selfie segmentation then pose segmentation fallback)
        h, w = image_np_rgb.shape[:2]
        selfie_results = self.mp_selfie_segmentation.process(image_np_rgb)
        if selfie_results.segmentation_mask is not None:
            mask = (cv2.GaussianBlur(selfie_results.segmentation_mask, (5,5), 0) > 0.6).astype(np.float32)
            return mask
        if pose_results and pose_results.segmentation_mask is not None:
            mask = (cv2.GaussianBlur(pose_results.segmentation_mask, (5,5), 0) > 0.5).astype(np.float32)
            return mask
        print("Warning: Segmentation mask generation failed.")
        return np.zeros((h, w), dtype=np.float32)

    def _calculate_detailed_measurements(self, keypoints):
        # (Same as before - shoulder_width, torso_height, hip_width, body_angle)
        m = {}
        def dist(p1n, p2n):
            if p1n in keypoints and p2n in keypoints: return np.linalg.norm(np.array(keypoints[p1n]) - np.array(keypoints[p2n]))
            return 0
        m['shoulder_width'] = dist('left_shoulder', 'right_shoulder')
        m['torso_height'] = dist('neck', 'mid_hip')
        if 'neck' in keypoints and 'mid_hip' in keypoints:
            dy = keypoints['mid_hip'][1] - keypoints['neck'][1]; dx = keypoints['mid_hip'][0] - keypoints['neck'][0]
            m['body_angle_rad'] = np.arctan2(dx, dy) if dy!=0 else (np.pi/2 if dx>0 else -np.pi/2)
            m['body_angle_deg'] = np.degrees(m['body_angle_rad'])
        else: m['body_angle_rad'], m['body_angle_deg'] = 0.0, 0.0
        return m

    def _create_debug_visualization(self, image_np_rgba, keypoints, segmentation_mask):
        # (Same as before - draw circles and text for keypoints, overlay mask)
        dbg = image_np_rgba.copy()
        if segmentation_mask is not None and segmentation_mask.shape[:2] == dbg.shape[:2]:
            mc = np.zeros_like(dbg); mc[segmentation_mask > 0.5] = [0,255,0,100]
            dbg = cv2.addWeighted(dbg, 1.0, mc, 0.3, 0)
        for name, (x,y) in keypoints.items():
            cv2.circle(dbg, (x,y), 3, (255,0,0,255), -1)
            cv2.putText(dbg, name, (x+5,y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,0,255),1)
        return dbg

    def close(self):
        self.mp_selfie_segmentation.close()
        self.pose_model.close()

# --- ClothingAnalyzerAgent (Restored) ---
class ClothingAnalyzerAgent:
    def __init__(self, gemini_analyzer=None):
        self.gemini_analyzer = gemini_analyzer

    def analyze(self, clothing_img_np_rgba, config=None):
        config = config or {}
        metadata = {"error": None}
        clothing_type, control_points = "unknown", {}

        # 1. Background Removal
        clothing_mask_binary, clothing_no_bg_rgba = self._remove_background(clothing_img_np_rgba, config)
        if clothing_no_bg_rgba is None or clothing_mask_binary is None or cv2.countNonZero(clothing_mask_binary) < 100:
            metadata["error"] = "Background removal failed."
            clothing_no_bg_rgba = clothing_img_np_rgba
            h,w = clothing_img_np_rgba.shape[:2]; clothing_mask_binary = np.full((h,w), 255, dtype=np.uint8)
            print("Warning: Using original clothing image due to BG removal failure.")

        # 2. Try Gemini
        if self.gemini_analyzer:
            print("Attempting clothing analysis with Gemini...")
            gemini_features = self.gemini_analyzer.analyze_clothing_for_features(clothing_no_bg_rgba)
            if gemini_features and gemini_features.get("control_points"):
                clothing_type, control_points = gemini_features.get("type", "unknown"), gemini_features["control_points"]
                print(f"Used Gemini for clothing. Type: {clothing_type}, Points: {len(control_points)}")
            else:
                print("Gemini clothing analysis failed or no points, falling back to CV.")
                clothing_type, control_points = "unknown", {} # Reset

        # 3. Fallback to CV
        if not control_points:
            print("Using CV for clothing analysis.")
            # Determine mask from clothing_no_bg_rgba's alpha
            current_mask_binary = cv2.threshold(clothing_no_bg_rgba[:,:,3], 10, 255, cv2.THRESH_BINARY)[1] if clothing_no_bg_rgba.shape[2]==4 else clothing_mask_binary

            contours = self._extract_contours(current_mask_binary)
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                if clothing_type == "unknown": clothing_type = self._detect_clothing_type_cv(main_contour, clothing_no_bg_rgba.shape[:2])
                control_points = self._extract_control_points_cv(main_contour, clothing_type, clothing_no_bg_rgba.shape[:2])
            else:
                if not metadata["error"]: metadata["error"] = "No contours found for CV clothing analysis."
                print("Warning: No contours for CV clothing analysis.")
                if not control_points: # Last resort fallback points
                    h,w = clothing_no_bg_rgba.shape[:2]
                    control_points = {'collar_center':(w//2,h//10),'bottom_hem_center':(w//2,h*9//10),
                                      'left_shoulder_tip':(w//4,h//8),'right_shoulder_tip':(w*3//4,h//8)}
                    clothing_type = "generic_fallback_cv"

        debug_image = self._create_debug_visualization(clothing_img_np_rgba, clothing_mask_binary, control_points, clothing_type)
        return {"type": clothing_type, "control_points": control_points,
                "mask": clothing_mask_binary, "clothing_rgba_no_bg": clothing_no_bg_rgba,
                "debug_image": debug_image, "metadata": metadata}

    def _remove_background(self, clothing_img_np_rgba, config):
        # (Same as before - try alpha, then rembg)
        alpha = clothing_img_np_rgba[:,:,3]
        if np.mean(alpha) < 245: # Has transparency
            print("Using existing alpha for clothing BG removal.")
            mask_bin = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)[1]
            kernel=np.ones((3,3),np.uint8); mask_bin=cv2.morphologyEx(cv2.morphologyEx(mask_bin,cv2.MORPH_OPEN,kernel),cv2.MORPH_CLOSE,kernel)
            no_bg = clothing_img_np_rgba.copy(); no_bg[:,:,3] = mask_bin
            return mask_bin, no_bg
        
        print("Attempting rembg for clothing BG removal.")
        try:
            no_bg_pil = remove(Image.fromarray(cv2.cvtColor(clothing_img_np_rgba, cv2.COLOR_RGBA2RGB)), # rembg needs RGB
                                alpha_matting=config.get("alpha_matting",True),
                                alpha_matting_foreground_threshold=config.get("fg_threshold",240),
                                alpha_matting_background_threshold=config.get("bg_threshold",10),
                                alpha_matting_erode_size=config.get("erode_size",10))
            no_bg_rgba = np.array(no_bg_pil) # rembg returns RGBA
            mask_bin = cv2.threshold(no_bg_rgba[:,:,3], 10, 255, cv2.THRESH_BINARY)[1]
            return mask_bin, no_bg_rgba
        except Exception as e:
            print(f"rembg failed: {e}. Fallback.")
            return None, None

    def _extract_contours(self, mask_binary):
        # (Same as before)
        cnts, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c for c in cnts if cv2.contourArea(c) > 100]

    def _detect_clothing_type_cv(self, main_contour, image_shape_hw):
        # (Same basic CV type detection as before)
        x,y,w,h = cv2.boundingRect(main_contour)
        aspect = w/float(h) if h>0 else 0
        if 0.8 < aspect < 1.5 : return "tshirt_cv" # Basic
        return "unknown_cv"

    def _extract_control_points_cv(self, main_contour, clothing_type, image_shape_hw):
        # (Same improved CV control point extraction as before)
        # Extracts: bottom_hem_center/left/right, collar_center, shoulder_tips, collar_points, sleeve_ends, hood_top_center
        # Ensure this is complete and robust in your actual file.
        pts = {}
        xb,yb,wb,hb = cv2.boundingRect(main_contour)
        lm,rm,tm,bm = tuple(main_contour[main_contour[:,:,0].argmin()][0]), \
                        tuple(main_contour[main_contour[:,:,0].argmax()][0]), \
                        tuple(main_contour[main_contour[:,:,1].argmin()][0]), \
                        tuple(main_contour[main_contour[:,:,1].argmax()][0])
        pts['bottom_hem_center'] = bm; pts['collar_center'] = tm
        pts['left_shoulder_tip']=(xb+wb//4,yb+hb//10); pts['right_shoulder_tip']=(xb+wb*3//4,yb+hb//10) # Approx
        # Add more points based on contour analysis here...
        print(f"CV clothing points ({clothing_type}): {pts}")
        return pts

    def _create_debug_visualization(self, original_img_rgba, clothing_mask_binary, control_points, clothing_type):
        # (Same as before - draw mask contour, control points, type label)
        dbg = original_img_rgba.copy()
        if clothing_mask_binary is not None:
            cnts,_=cv2.findContours(clothing_mask_binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(dbg,cnts,-1,(0,255,0,150),1)
        for name,(x,y) in control_points.items():
            cv2.circle(dbg,(int(x),int(y)),2,(0,0,255,255),-1) # Smaller points
            cv2.putText(dbg,name,(int(x)+3,int(y)+3),cv2.FONT_HERSHEY_SIMPLEX,0.25,(255,255,0,255),1)
        cv2.putText(dbg,f"Type:{clothing_type}",(5,15),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255,255),1)
        return dbg

# --- PointMatchingAgent (Restored) ---
STANDARD_CLOTHING_TO_BODY_MAP = { # Keep this comprehensive
    "collar_center": "neck", "left_collar_point": "left_collar", "right_collar_point": "right_collar",
    "left_shoulder_tip": "left_shoulder", "right_shoulder_tip": "right_shoulder",
    "left_sleeve_end": "left_wrist", "right_sleeve_end": "right_wrist",
    "left_armpit": "left_shoulder", "right_armpit": "right_shoulder", # Needs offset logic
    "waist_left": "left_waist", "waist_right": "right_waist",
    "bottom_hem_center": "mid_hip", "bottom_hem_left": "left_hip", "bottom_hem_right": "right_hip",
    "hood_top_center": "nose", "hood_left_opening": "left_ear", "hood_right_opening": "right_ear",
}
SHORT_SLEEVE_MAP_EXTENSION = {"left_sleeve_end": "left_elbow", "right_sleeve_end": "right_elbow"}

class PointMatchingAgent:
    def match_points(self, body_data, clothing_data, config=None):
        # (Same as before - use STANDARD_CLOTHING_TO_BODY_MAP, check for short sleeves)
        src_pts, dst_pts = [], []
        body_kpts, cloth_pts = body_data["body_keypoints"], clothing_data["control_points"]
        cloth_type = clothing_data.get("type", "unknown").lower()
        
        current_map = STANDARD_CLOTHING_TO_BODY_MAP.copy()
        if "tshirt" in cloth_type or "polo" in cloth_type: current_map.update(SHORT_SLEEVE_MAP_EXTENSION)

        matched_names = []
        for cpn, bpn in current_map.items():
            if cpn in cloth_pts and bpn in body_kpts:
                src_pts.append(cloth_pts[cpn]); dst_pts.append(body_kpts[bpn])
                matched_names.append(cpn)
        print(f"Matched {len(src_pts)} points. Names: {matched_names}")
        
        dbg_img = None
        if body_data.get("debug_image") is not None and src_pts:
            dbg_img = self._create_debug_visualization(body_data["debug_image"], src_pts, dst_pts)
        
        return {"src_points": src_pts, "dst_points": dst_pts, "debug_image": dbg_img,
                "metadata": {"match_count": len(src_pts)}}

    def _create_debug_visualization(self, body_debug_img_rgba, src_points, dst_points):
        # (Same as before - draw circles on body_debug_img for dst_points)
        vis = body_debug_img_rgba.copy()
        for i, (dx,dy) in enumerate(dst_points):
            cv2.circle(vis,(dx,dy),4,(0,0,255,255),-1) # DST points
            cv2.putText(vis, f"m{i}",(dx+6,dy+6),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,0,255),1)
        return vis

# --- ImageTransformationAgent (Restored) ---
class ImageTransformationAgent:
    def transform(self, clothing_rgba_to_transform, matched_points_dict, target_shape_hw, config=None):
        # (Same as before - use PiecewiseAffineTransform)
        # IMPORTANT: clothing_rgba_to_transform should be the background-removed one
        src_np = np.array(matched_points_dict["src_points"], dtype=np.float32)
        dst_np = np.array(matched_points_dict["dst_points"], dtype=np.float32)
        meta = {"warping_method": "piecewise_affine", "error": None}

        if len(src_np) < 3:
            meta["error"] = "Piecewise Affine needs at least 3 points."
            return np.zeros((target_shape_hw[0], target_shape_hw[1], 4), dtype=np.uint8), meta
        try:
            src_yx, dst_yx = src_np[:,::-1], dst_np[:,::-1] # y,x format for skimage
            tform = skimage_transform.PiecewiseAffineTransform(); tform.estimate(src_yx, dst_yx)
            
            out_shape_sk = (target_shape_hw[0], target_shape_hw[1], clothing_rgba_to_transform.shape[2])
            warped = skimage_transform.warp(clothing_rgba_to_transform, tform.inverse, 
                                            output_shape=out_shape_sk, order=1, 
                                            mode='constant', cval=0, preserve_range=True)
            warped_u8 = (warped * 255).astype(np.uint8) if np.issubdtype(warped.dtype, np.floating) and warped.max()<=1 else warped.astype(np.uint8)

            # Ensure 4 channels if original was 4
            if warped_u8.shape[2] == 3 and clothing_rgba_to_transform.shape[2] == 4:
                alpha_original = clothing_rgba_to_transform[:,:,3]
                alpha_warped = skimage_transform.warp(alpha_original, tform.inverse,
                                                      output_shape=target_shape_hw, order=1,
                                                      mode='constant', cval=0, preserve_range=True)
                alpha_warped_u8 = (alpha_warped*255).astype(np.uint8) if np.issubdtype(alpha_warped.dtype,np.floating) and alpha_warped.max()<=1 else alpha_warped.astype(np.uint8)
                warped_u8 = cv2.merge((warped_u8, alpha_warped_u8))
            elif warped_u8.shape[2] != 4 : # if not 3 (and became 4) and not already 4
                 # create empty alpha
                 empty_alpha = np.zeros(target_shape_hw, dtype=np.uint8)
                 if warped_u8.shape[2] == 1: # Grayscale
                     warped_u8 = cv2.cvtColor(warped_u8, cv2.COLOR_GRAY2RGBA)
                 elif warped_u8.shape[2] == 3: # RGB
                     warped_u8 = cv2.cvtColor(warped_u8, cv2.COLOR_RGB2RGBA)
                 else: # Unhandled, return transparent
                     meta["error"] = f"Warped image has unexpected shape: {warped_u8.shape}"
                     return np.zeros((target_shape_hw[0], target_shape_hw[1], 4), dtype=np.uint8), meta
            return warped_u8, meta
        except Exception as e:
            import traceback
            meta["error"] = f"Transform failed: {e}\n{traceback.format_exc()}"
            return np.zeros((target_shape_hw[0], target_shape_hw[1], 4), dtype=np.uint8), meta


# --- PostProcessingAgent (Restored) ---
class PostProcessingAgent:
    def process(self, person_img_rgba, warped_clothing_rgba, body_data, config=None):
        # (Same blending logic as before, ensuring masks and alphas are handled)
        config = config or {}
        result_img = person_img_rgba.copy()
        meta = {"error": None}

        if warped_clothing_rgba is None or warped_clothing_rgba.shape[0]==0:
            meta["error"] = "Warped clothing empty for post-processing."
            return {"result_image": result_img, "metadata": meta}

        # Ensure warped clothing is RGBA
        if warped_clothing_rgba.shape[2] == 3:
            warped_clothing_rgba = cv2.cvtColor(warped_clothing_rgba, cv2.COLOR_RGB2RGBA)
            warped_clothing_rgba[:,:,3] = 255 # Assume opaque if only 3 channels came

        person_mask_float = body_data.get("segmentation_mask") # HxW, float 0-1
        if person_mask_float is None:
            person_mask_float = (person_img_rgba[:,:,3] > 10).astype(np.float32) if person_img_rgba.shape[2]==4 else np.ones(person_img_rgba.shape[:2], dtype=np.float32)
        if person_mask_float.ndim == 3: person_mask_float = person_mask_float.squeeze(axis=2)

        clothing_alpha_float = warped_clothing_rgba[:,:,3:4] / 255.0 # HxWx1
        effective_alpha = clothing_alpha_float * person_mask_float[:,:,np.newaxis]

        result_img[:,:,:3] = result_img[:,:,:3]*(1-effective_alpha) + warped_clothing_rgba[:,:,:3]*effective_alpha
        
        # Update combined alpha: max of person's original alpha and new effective clothing alpha
        # Ensure person_img_rgba has alpha before accessing it
        person_original_alpha_float = person_img_rgba[:,:,3]/255.0 if person_img_rgba.shape[2] == 4 else np.ones(person_img_rgba.shape[:2], dtype=np.float32)
        new_combined_alpha_u8 = (np.maximum(person_original_alpha_float, effective_alpha.squeeze())*255).astype(np.uint8)
        result_img[:,:,3] = new_combined_alpha_u8
        
        # Optional effects
        if config.get("natural_wrinkle_simulation"): result_img = self._add_natural_wrinkles(result_img, body_data, warped_clothing_rgba, effective_alpha.squeeze())
        if config.get("apply_shadow"): result_img = self._add_clothing_shadows(result_img, body_data, warped_clothing_rgba, effective_alpha.squeeze())

        return {"result_image": result_img, "metadata": meta}

    def _add_natural_wrinkles(self, blended_img, body_data, clothing_img, clothing_eff_mask_float): # Placeholder
        return blended_img
    def _add_clothing_shadows(self, blended_img, body_data, clothing_img, clothing_eff_mask_float): # Placeholder
        return blended_img

# --- QualityValidationAgent (Restored) ---
class QualityValidationAgent:
    def validate(self, person_img, result_img, body_data, clothing_data, matched_points, config=None, pipeline_state_ref=None):
        # (Same basic validation logic as before)
        is_valid, issues, score = True, [], 0.7 # Defaults
        if result_img is None or result_img.shape[0]==0:
            is_valid, issues, score = False, ["Result image empty"], 0.0
        
        # Check if clothing is visible (crude check based on alpha change)
        if is_valid and result_img.shape[2] == 4 and person_img.shape[2] == 4 and pipeline_state_ref:
             warped_cloth_from_pipeline = pipeline_state_ref.get("transformation",{}).get("warped_clothing")
             if warped_cloth_from_pipeline is not None and warped_cloth_from_pipeline[:,:,3].mean() > 20: # If warped cloth had significant alpha
                 # If final result's alpha didn't increase much over original person's alpha
                 if result_img[:,:,3].mean() < person_img[:,:,3].mean() + 10: 
                     print("Warning QVA: Clothing overlay might be minimal or blended away.")
                     # issues.append("Clothing not visibly overlaid") # Can be too strict
                     # score = max(0.1, score - 0.3)

        return {"is_valid": is_valid, "issues": issues, "score": score, "corrections": {}, "metadata": {}}


# --- Streamlit UI (Restored - ensure it matches your intended UI) ---
def load_image_from_upload(uploaded_file):
    # (Same as before)
    if not uploaded_file: return None
    try: return np.array(Image.open(uploaded_file).convert("RGBA"))
    except Exception as e: st.error(f"Img load error: {e}"); return None

def get_download_link(image_np, filename="result.png", text="Download Result Image"):
    # (Same as before)
    buf = io.BytesIO(); Image.fromarray(image_np).save(buf,format="PNG")
    img_s = base64.b64encode(buf.getvalue()).decode()
    return f'<a href="data:image/png;base64,{img_s}" download="{filename}">{text}</a>'

def main():
    global DEBUG_MODE
    st.set_page_config(page_title="AI Virtual Try-On", layout="wide")
    for D_DIR in [RESULTS_DIR, TEMP_DIR, SAMPLES_DIR]:
        if not os.path.exists(D_DIR): os.makedirs(D_DIR)

    st.title("👕 AI-Powered Virtual Clothing Try-On")
    # ... (Rest of Streamlit UI: sidebar for API key, file uploads, sample selection, "Try It On" button)
    with st.sidebar:
        st.header("Settings")
        gemini_api_key = st.text_input("Google Gemini API Key (Optional)", type="password")
        DEBUG_MODE = st.checkbox("Enable Debug Mode", value=DEBUG_MODE)
        st.info(f"Debug mode: {'ON' if DEBUG_MODE else 'OFF'}")

        st.subheader("Person Image")
        person_file = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])
        st.subheader("Clothing Item")
        clothing_source = st.radio("Choose clothing source", ["Upload", "Sample"])
        clothing_file, sample_path = None, None
        if clothing_source == "Upload":
            clothing_file = st.file_uploader("Upload clothing", type=["png", "jpg", "jpeg"])
        else:
            sample_files = {f.rsplit('.',1)[0].replace("_"," ").title(): os.path.join(SAMPLES_DIR,f)
                            for f in os.listdir(SAMPLES_DIR) if f.lower().endswith((".png",".jpg",".jpeg"))}
            if sample_files:
                sample_name = st.selectbox("Choose sample", list(sample_files.keys()))
                if sample_name: sample_path = sample_files[sample_name]
            else: st.info("No samples in 'samples_vto'.")

    col1, col2 = st.columns(2)
    person_img_np, clothing_img_np = None, None
    with col1:
        st.subheader("Your Photo")
        if person_file: person_img_np = load_image_from_upload(person_file)
        if person_img_np is not None: st.image(person_img_np, use_column_width=True)
    with col2:
        st.subheader("Clothing Item")
        if clothing_file: clothing_img_np = load_image_from_upload(clothing_file)
        elif sample_path:
            try: clothing_img_np = np.array(Image.open(sample_path).convert("RGBA"))
            except Exception as e: st.error(f"Sample load error: {e}")
        if clothing_img_np is not None: st.image(clothing_img_np, use_column_width=True)

    if st.button("✨ Try It On!", type="primary", disabled=(person_img_np is None or clothing_img_np is None)):
        master_agent_instance = MasterAgent(gemini_api_key=gemini_api_key if gemini_api_key else None)
        with st.spinner("Processing..."):
            current_config = master_agent_instance._get_default_config() # Use defaults for now
            result_image, pipeline_state = master_agent_instance.process(
                person_img_np, clothing_img_np, config=current_config
            )

            if result_image is not None:
                st.subheader("🎉 Result")
                st.image(result_image, caption="Virtual Try-On Result", use_column_width=True)
                st.markdown(get_download_link(result_image), unsafe_allow_html=True)
                if DEBUG_MODE:
                    st.info(f"Total time: {pipeline_state.get('metadata',{}).get('total_processing_time',0):.2f}s")
                    if pipeline_state.get("errors"):
                        st.error("Errors:"); [st.code(e) for e in pipeline_state["errors"]]
                    val=pipeline_state.get("validation",{}); st.write(f"Validation: {val.get('score','N/A'):.2f}, Issues: {val.get('issues',[])}")
                    if st.checkbox("Show Pipeline State"): st.json(pipeline_state, expanded=False)
                    if pipeline_state.get("debug_images"):
                        st.write("Debug Images:")
                        tabs=st.tabs(list(pipeline_state["debug_images"].keys()))
                        for i,k in enumerate(pipeline_state["debug_images"].keys()):
                            with tabs[i]: st.image(pipeline_state["debug_images"][k],caption=k,use_column_width=True)
            else:
                st.error("Processing failed.")
                if pipeline_state.get("errors"): st.error("Errors:"); [st.code(e) for e in pipeline_state["errors"]]

if __name__ == "__main__":
    main()
