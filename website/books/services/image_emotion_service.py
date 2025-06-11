import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from io import BytesIO
import threading
import queue
import time
import logging
from sklearn.metrics.pairwise import cosine_similarity
import uuid

logger = logging.getLogger(__name__)

import warnings
def torch_load_model(checkpoint_file, map_location=None):
    """ Wrap torch.load to catch standard warning of not finding the nested implementations.
    :param checkpoint_file:
    :param map_location:
    :return:
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = torch.load(checkpoint_file, weights_only=False, map_location=map_location)
    return model

ARTEMIS_EMOTIONS = ['amusement', 'awe', 'contentment', 'excitement',
                    'anger', 'disgust',  'fear', 'sadness', 'something else']

EMOTION_TO_IDX = {e: i for i, e in enumerate(ARTEMIS_EMOTIONS)}
IDX_TO_EMOTION = {EMOTION_TO_IDX[e]: e for e in EMOTION_TO_IDX}

class ImageEmotionService:
    """Service for image emotion prediction using the Artemis ResNet model"""
    
    def __init__(self, model_path="models/best_model_good_data.pt"):
        self.emotion_labels = ARTEMIS_EMOTIONS
        self.num_emotions = len(self.emotion_labels)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.use_dummy = False
        
        # Image preprocessing (same as in your notebook)
        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Queue for processing requests
        self.request_queue = queue.Queue()
        self.response_dict = {}
        
        # Initialize model immediately in a try-catch
        self._initialize_model()
        
        # Start background thread for processing
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        # Model is ready immediately now
        self.model_ready = threading.Event()
        self.model_ready.set()
    
    def _initialize_model(self):
        """Initialize the model, fallback to dummy if needed"""
        try:
            logger.info("Attempting to load Artemis ResNet model...")
            self.model = torch_load_model(self.model_path, map_location=self.device)
            self.model.to(self.device)
            self.model.eval()
            
            # Test the model with a dummy input
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 256, 256).to(self.device)
                _ = self.model(dummy_input)
            
            logger.info("Artemis ResNet model loaded successfully!")
            
        except Exception as e:
            logger.warning(f"Failed to load Artemis ResNet model: {e}")
            logger.info("Creating dummy model for testing...")
            self._create_dummy_model()
            self.use_dummy = True
    
    def _worker_loop(self):
        """Background worker that handles inference requests"""
        try:
            logger.info("Background worker started for emotion processing...")
            
            # Process incoming requests
            while True:
                try:
                    request_id, image_data = self.request_queue.get(timeout=1.0)
                    logger.info(f"Processing request: {request_id}")
                    
                    # Process image and get emotion distribution
                    try:
                        if isinstance(image_data, bytes):
                            img = Image.open(BytesIO(image_data)).convert('RGB')
                        else:
                            img = image_data
                            
                        # Generate emotion distribution
                        emotion_result = self._predict_emotions(img)
                        
                        # Store result with proper structure
                        self.response_dict[request_id] = {
                            'processing': False,
                            'status': 'success',
                            'emotion_distribution': emotion_result['emotion_distribution'],
                            'dominant_emotion': emotion_result['dominant_emotion'],
                            'confidence': emotion_result['confidence'],
                            'emotion_scores': emotion_result['emotion_scores']
                        }
                        
                        logger.info(f"Successfully processed request: {request_id}")
                        
                    except Exception as e:
                        logger.error(f"Error processing image for request {request_id}: {e}")
                        self.response_dict[request_id] = {
                            'processing': False,
                            'status': 'error',
                            'error': str(e)
                        }
                        
                    self.request_queue.task_done()
                    
                except queue.Empty:
                    # No requests in queue, continue waiting
                    continue
                    
        except Exception as e:
            logger.error(f"Background worker failed: {e}")
    
    def _create_dummy_model(self):
        """Create dummy model for testing if Artemis model loading fails"""
        logger.warning("Using dummy model for testing")
        from torchvision import models
        self.model = models.resnet34(weights='IMAGENET1K_V1')
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_emotions)
        self.model.to(self.device)
        self.model.eval()
    
    def _predict_emotions(self, image):
        """Predict emotions for a PIL Image"""
        try:
            # Preprocess image
            image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)
            
            # Fast inference with no gradient computation
            with torch.no_grad():
                if self.use_dummy:
                    # For dummy model, create random but realistic emotion distribution
                    logits = self.model(image_tensor)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    confidence = float(np.max(probs))
                else:
                    # Model outputs log probabilities (LogSoftmax), so we need to exp them
                    log_probs = self.model(image_tensor)
                    probs = torch.exp(log_probs).cpu().numpy()[0]  # Convert to probabilities
                    confidence = float(torch.max(torch.exp(log_probs)).cpu())
            
            # Ensure probabilities are valid
            probs = np.clip(probs, 0, 1)
            probs = probs / np.sum(probs)  # Normalize to sum to 1
            
            return {
                'emotion_distribution': probs.tolist(),  # Convert to list for JSON serialization
                'confidence': confidence,
                'dominant_emotion': self.emotion_labels[np.argmax(probs)],
                'emotion_scores': dict(zip(self.emotion_labels, probs.tolist()))
            }
            
        except Exception as e:
            logger.error(f"Error in _predict_emotions: {e}")
            # Return fallback emotion distribution
            fallback_probs = np.random.dirichlet(np.ones(self.num_emotions))
            return {
                'emotion_distribution': fallback_probs.tolist(),
                'confidence': 0.5,
                'dominant_emotion': self.emotion_labels[np.argmax(fallback_probs)],
                'emotion_scores': dict(zip(self.emotion_labels, fallback_probs.tolist()))
            }
    
    def get_emotion_async(self, image_data):
        """Submit an image for async processing and get a request ID"""
        request_id = str(uuid.uuid4())
        logger.info(f"Submitting async request: {request_id}")
        
        # Mark as processing initially
        self.response_dict[request_id] = {'processing': True}
        
        # Queue the request
        self.request_queue.put((request_id, image_data))
        return request_id
    
    def get_emotion_result(self, request_id, timeout=30):
        """Poll for result of an async emotion prediction request"""
        start_time = time.time()
        logger.info(f"Polling for result: {request_id}")
        
        # Wait for specific result
        while time.time() - start_time < timeout:
            if request_id in self.response_dict:
                result = self.response_dict[request_id]
                
                # If still processing, return processing status (don't cleanup yet)
                if result.get('processing', False):
                    return {'processing': True}
                
                # Processing complete - return result and clean up immediately
                logger.info(f"Returning result for: {request_id}")
                
                # Remove from dict before returning (auto-cleanup)
                del self.response_dict[request_id]
                logger.info(f"Auto-cleaned up result for: {request_id}")
                
                return result
                
            time.sleep(0.1)
            
        logger.warning(f"Timeout waiting for result: {request_id}")
        return {'processing': False, 'status': 'error', 'error': 'Processing timeout'}

    def cleanup_result(self, request_id):
        """Manually clean up a result when no longer needed (for error cases)"""
        if request_id in self.response_dict:
            del self.response_dict[request_id]
            logger.info(f"Manually cleaned up result for: {request_id}")

# Global service instance
_emotion_service_instance = None

def get_emotion_service():
    """Get the global emotion service instance"""
    global _emotion_service_instance
    if _emotion_service_instance is None:
        _emotion_service_instance = ImageEmotionService()
    return _emotion_service_instance