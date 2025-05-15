import torch
import os
import threading
import queue
import time
from PIL import Image
from torchvision import transforms
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

class ImageEmbeddingService:
    """Service that handles image embedding in a background thread to prevent web worker timeouts"""
    
    def __init__(self, model_path="models/EmotionCLIP/emotionclip_latest.pt"):
        self.model_path = model_path
        self.model = None
        self.preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073], 
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        self.dummy_mask = torch.ones((1, 224, 224), dtype=torch.bool)
        
        # Queue for processing requests
        self.request_queue = queue.Queue()
        self.response_dict = {}
        
        # Start background thread for model loading and processing
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        # Wait for model to load on first request
        self.model_ready = threading.Event()
    
    def _worker_loop(self):
        """Background worker that handles model loading and inference"""
        try:
            # Import EmotionCLIP in the background thread
            logger.info("Starting model loading in background thread...")
            from models.EmotionCLIP.src.models.base import EmotionCLIP
            
            # Load the model
            try:
                checkpoint = torch.load(self.model_path, map_location="cpu")
                
                if "state_dict" in checkpoint:
                    hparams = checkpoint.get("hyper_parameters", {})
                    self.model = EmotionCLIP(**hparams)
                    state = checkpoint["state_dict"]
                else:
                    self.model = EmotionCLIP()
                    state = checkpoint
                    
                self.model.load_state_dict(state, strict=False)
                self.model.eval()
                logger.info("Model loaded successfully in background thread!")
                self.model_ready.set()  # Signal that model is ready
                
            except Exception as e:
                logger.error(f"Failed to load EmotionCLIP model: {e}")
                return
            
            # Process incoming requests
            while True:
                try:
                    request_id, image_data = self.request_queue.get(timeout=1.0)
                    
                    # Process image and get embedding
                    try:
                        if isinstance(image_data, bytes):
                            img = Image.open(BytesIO(image_data)).convert('RGB')
                        else:
                            img = image_data
                            
                        # Generate embedding
                        with torch.no_grad():
                            x = self.preprocess(img).unsqueeze(0)
                            embedding = self.model.encode_image(x, self.dummy_mask)
                            embedding = torch.nn.functional.normalize(embedding, dim=-1)
                            result = embedding.squeeze(0).tolist()
                            
                        # Store result
                        self.response_dict[request_id] = {
                            'status': 'success',
                            'embedding': result
                        }
                        
                    except Exception as e:
                        logger.error(f"Error processing image: {e}")
                        self.response_dict[request_id] = {
                            'status': 'error',
                            'error': str(e)
                        }
                        
                    self.request_queue.task_done()
                    
                except queue.Empty:
                    # No requests in queue, continue waiting
                    continue
                    
        except Exception as e:
            logger.error(f"Background worker failed: {e}")
    
    def get_embedding_async(self, image_data):
        """Submit an image for async processing and get a request ID"""
        request_id = f"req_{time.time()}_{hash(str(image_data))}"
        self.request_queue.put((request_id, image_data))
        return request_id
    
    def get_embedding_result(self, request_id, timeout=30):
        """Poll for result of an async embedding request"""
        start_time = time.time()
        
        # First wait for model to be loaded
        if not self.model_ready.wait(timeout=timeout):
            return {'status': 'error', 'error': 'Model loading timeout'}
        
        # Then wait for specific result
        while time.time() - start_time < timeout:
            if request_id in self.response_dict:
                result = self.response_dict[request_id]
                del self.response_dict[request_id]  # Clean up
                return result
            time.sleep(0.1)
            
        return {'status': 'error', 'error': 'Processing timeout'}

# Global service instance
_service_instance = None

def get_embedding_service():
    """Get the global embedding service instance"""
    global _service_instance
    if _service_instance is None:
        _service_instance = ImageEmbeddingService()
    return _service_instance