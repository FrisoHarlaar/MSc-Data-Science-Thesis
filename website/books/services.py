import torch
import queue
import threading
import logging
from PIL import Image
from io import BytesIO
from torchvision import transforms
import uuid

logger = logging.getLogger(__name__)

class ImageEmbeddingService:
    """Service that handles image embedding in a background thread to prevent web worker timeouts"""
    
    def __init__(self, model_path="models/EmotionCLIP/emotionclip_latest.pt"):
        self.model_path = model_path
        self.model = None
        
        # Set device - prioritize GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"ImageEmbeddingService using device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        self.preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073], 
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        self.dummy_mask = torch.ones((1, 224, 224), dtype=torch.bool, device=self.device)
        
        # Queue for processing requests
        self.request_queue = queue.Queue()
        self.response_dict = {}
        self.model_ready = threading.Event()
        
        # Start background thread for model loading and processing
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _worker_loop(self):
        """Background worker that handles model loading and inference"""
        try:
            # Import EmotionCLIP in the background thread
            logger.info("Starting model loading in background thread...")
            from models.EmotionCLIP.src.models.base import EmotionCLIP
            
            # Load the model
            try:
                # Load checkpoint to CPU first, then move to device
                checkpoint = torch.load(self.model_path, map_location="cpu")
                
                if "state_dict" in checkpoint:
                    hparams = checkpoint.get("hyper_parameters", {})
                    self.model = EmotionCLIP(**hparams)
                    state = checkpoint["state_dict"]
                else:
                    self.model = EmotionCLIP()
                    state = checkpoint
                    
                self.model.load_state_dict(state, strict=False)
                
                # Move model to GPU and set to eval mode
                self.model = self.model.to(self.device)
                self.model.eval()
                
                logger.info(f"Model loaded successfully on {self.device}!")
                
                # Warm up the model with a dummy forward pass
                if self.device.type == 'cuda':
                    dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
                    dummy_mask = torch.ones((1, 224, 224), dtype=torch.bool, device=self.device)
                    with torch.no_grad():
                        _ = self.model.encode_image(dummy_input, dummy_mask)
                    logger.info("GPU model warmed up successfully!")
                
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
                            
                        # Generate embedding on GPU
                        with torch.no_grad():
                            x = self.preprocess(img).unsqueeze(0).to(self.device)  # Move input to GPU
                            embedding = self.model.encode_image(x, self.dummy_mask)
                            embedding = torch.nn.functional.normalize(embedding, dim=-1)
                            # Move result back to CPU for storage
                            result = embedding.cpu().squeeze(0).tolist()
                            
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
                    continue
                except Exception as e:
                    logger.error(f"Error in worker loop: {e}")
                    
        except Exception as e:
            logger.error(f"Critical error in worker thread: {e}")

    def get_embedding_async(self, image_data):
        """Submit an image for async processing and get a request ID"""
        request_id = str(uuid.uuid4())
        self.request_queue.put((request_id, image_data))
        return request_id

    def get_embedding_result(self, request_id, timeout=30):
        """Poll for result of an async embedding request"""
        # Wait for model to be ready first
        if not self.model_ready.wait(timeout=timeout):
            return {'status': 'error', 'error': 'Model loading timeout'}
        
        # Check if result is available
        if request_id in self.response_dict:
            result = self.response_dict.pop(request_id)
            return result
        else:
            return {'status': 'pending'}